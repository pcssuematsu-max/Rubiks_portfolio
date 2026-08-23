"""Learning orchestration helpers."""

from __future__ import annotations

import gc
import os
import resource
import subprocess
import sys
from collections import Counter

class LearnManager:
    """FrameからAI学習の実行順序とSearch2/Search3の呼び分けを切り出す。"""

    def __init__(self, frame):
        self.frame = frame

    def learn_all(self):
        """登録されている全AIを学習する。"""
        self.learn_indices(range(self.frame.AInum))

    def learn_indices(self, indices):
        """指定されたAI index群だけを順番に学習する。"""
        for index in indices:
            if self.should_learn(index):
                result = self.learn_one(index)
                self.log_result(index,result)
        self.prune_training_data()
        self.release_memory()
        self.frame.append_log(self.memory_status_text())

    def should_learn(self, index):
        """指定AIを学習対象にするか判定する。現状は既存条件をそのまま維持する。"""
        return len(self.frame.AIs[index].indices) >= 0

    def learn_one(self, index):
        """1つのAIを学習し、学習後処理を行って結果を返す。"""
        ai = self.frame.AIs[index]
        self.frame.append_log(f'AI {index} learn start ({ai.search_mode})')
        result = self.run_learning(index,ai)
        self.after_learning(ai)
        return result

    def run_learning(self, index, ai):
        """AIのsearch_modeに応じてSearch2型またはSearch3型の学習を呼び分ける。"""
        if self.uses_search2_learning(ai):
            return ai.learn(
                transformation = self.transformation_for(index),
                flip_inside = self.flip_inside_for(index),
                progress_callback = lambda message: self.log_progress(index, message),
            )

        return ai.learn_search3(
            transformation = self.transformation_for(index),
            flip_inside = self.flip_inside_for(index),
            progress_callback = lambda message: self.log_progress(index, message),
        )

    def uses_search2_learning(self, ai):
        """このAIがSearch2型の学習を使うか判定する。"""
        return ai.search_mode == 'search2'

    def transformation_for(self, index):
        """指定AIに対応するキューブ変換indexを返す。"""
        return self.frame.transform_idx[index]

    def flip_inside_for(self, index):
        """指定AIに対応する内側反転フラグを返す。"""
        return self.frame.flip_inside_idx[index]

    def after_learning(self, ai):
        """学習後に完成状態の評価値を再計算する。"""
        ai.set_perfect_val()

    def prune_training_data(self):
        """古い学習データを上限内に保ち、参照 index を補正する。"""
        search2_pruned = self.prune_search2_data()
        search3_pruned = self.prune_search3_data()
        if search2_pruned > 0 or search3_pruned > 0:
            self.frame.data_len = len(self.frame.AIs[0].datas)
            self.frame.data_search3_len = self.frame.current_search3_data_len()
            self.frame.append_log(
                f'pruned training data: search2={search2_pruned} search3={search3_pruned}'
            )

    def prune_search2_data(self):
        """Search2共有データを末尾側だけ残す。"""
        limit = int(getattr(self.frame.config,'max_search2_data',0) or 0)
        if limit <= 0 or not self.frame.AIs:
            return 0
        datas = self.frame.AIs[0].datas
        excess = len(datas) - limit
        if excess <= 0:
            return 0

        del datas[:excess]
        for ai in self.frame.AIs:
            ai.indices = [data_index - excess for data_index in ai.indices if data_index >= excess]
        if hasattr(self.frame,'myval_AI') and self.frame.myval_AI not in self.frame.AIs:
            self.frame.myval_AI.indices = [
                data_index - excess
                for data_index in self.frame.myval_AI.indices
                if data_index >= excess
            ]
        return excess

    def prune_search3_data(self):
        """Search3データをAIごとに末尾側だけ残す。"""
        limit = int(getattr(self.frame.config,'max_search3_data_per_ai',0) or 0)
        if limit <= 0:
            return 0
        pruned = 0
        for ai in self.frame.AIs:
            excess = len(ai.datas_search3) - limit
            if excess <= 0:
                continue
            del ai.datas_search3[:excess]
            ai.indices_search3 = [
                data_index - excess
                for data_index in ai.indices_search3
                if data_index >= excess
            ]
            pruned += excess
        return pruned

    def release_memory(self):
        """Python/MPS/CUDAの解放可能なキャッシュを掃除する。"""
        gc.collect()
        for ai in self.frame.AIs:
            if hasattr(ai,'_mark_torch_params_dirty'):
                ai._mark_torch_params_dirty()
            if hasattr(ai,'clear_training_cache'):
                ai.clear_training_cache(collect = False)
            if hasattr(ai,'_release_torch_training_memory'):
                ai._release_torch_training_memory()

    def memory_status_text(self):
        """学習後にメモリ関連の保持数を短く表示する。"""
        rss_mb = self.current_rss_mb()
        torch_cached = sum(
            1
            for ai in self.frame.AIs
            if getattr(ai,'_torch_params_cache',None) is not None
        )
        training_devices = Counter(str(getattr(ai,'torch_training_device','auto')) for ai in self.frame.AIs)
        torch_flags = Counter(bool(getattr(ai,'use_torch',False)) for ai in self.frame.AIs)
        torch_predict_flags = Counter(bool(getattr(ai,'use_torch_predict',False)) for ai in self.frame.AIs)
        torch_training_flags = Counter(bool(getattr(ai,'use_torch_training',False)) for ai in self.frame.AIs)
        torch_memory = self.torch_memory_status()
        log_lines = getattr(getattr(self.frame,'log_viewer',None),'line_count',None)
        return (
            f'memory status: rss={rss_mb:.1f}MB '
            f'search2_data={len(self.frame.AIs[0].datas)} '
            f'search3_data={self.frame.current_search3_data_len()} '
            f'torch_param_caches={torch_cached} '
            f'use_torch={dict(torch_flags)} '
            f'torch_predict={dict(torch_predict_flags)} '
            f'torch_training={dict(torch_training_flags)} '
            f'train_devices={dict(training_devices)} '
            f'torch_memory={torch_memory} '
            f'log_lines={log_lines}'
        )

    def torch_memory_status(self):
        for ai in self.frame.AIs:
            if hasattr(ai,'torch_memory_status'):
                return ai.torch_memory_status()
        return {}

    def current_rss_mb(self):
        """現在RSSをpsから取得し、失敗時は最大RSSへfallbackする。"""
        try:
            output = subprocess.check_output(
                ['ps','-o','rss=','-p',str(os.getpid())],
                text = True,
            )
            return float(output.strip()) / 1024.0
        except Exception:
            max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform == 'darwin':
                return max_rss / (1024.0 * 1024.0)
            return max_rss / 1024.0

    def log_result(self, index, result):
        """学習結果を GUI ログへ表示する。"""
        self.frame.append_log(f'AI {index} result: {result}')

    def log_progress(self, index, message):
        """学習途中の進捗を GUI ログへ表示する。"""
        self.frame.append_log(f'AI {index}: {message}')
