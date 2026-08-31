"""Top control panel widget."""

import tkinter as Tk

class ControlPanel(Tk.Frame):
    """Frame上部の操作ボタン・入力欄をまとめて配置する。"""

    GRAD_MODE_CHOICES = (
        ('W1', 'W1 の重み'),
        ('SVD', 'SVD'),
        ('Grad', '重要度（Gradient）'),
        ('IG', '重要度（Integrated Gradients）'),
        ('Contrast', '比較（Contrast）'),
        ('Occ', '隠したときの影響（Occlusion）'),
        ('PieceOcc', 'ピース単位の影響（Occlusion）'),
        ('PolicyOcc', '手の選択への影響（Occlusion）'),
        ('PiecePolicyOcc', 'ピースごとの手への影響'),
        ('AttnIn', '入力への注目度（Attention）'),
        ('AttnOut', '出力への注目度（Attention）'),
        ('AttnCentral', '中央への注目度（Attention）'),
        ('EmbNorm', 'Embedding の大きさ'),
        ('EmbPC1', 'Embedding の第1主成分'),
    )

    def __init__(self,master,frame,initial_mode = 'advanced'):
        Tk.Frame.__init__(self,master,relief = Tk.RIDGE,bd = 4)
        self.frame = frame
        self.font = ('Century Gothic',12,'bold')
        self.panel_mode = initial_mode
        self._build_buttons()
        self._configure_columns()

    def _build_buttons(self):
        """操作ボタンと入力欄を生成し、grid上に配置する。"""
        self._build_mode_controls()
        self.simple_controls = Tk.Frame(self)
        self.simple_controls.grid(row = 1,column = 0,sticky = 'ew')
        self.advanced_controls = Tk.LabelFrame(self,text = 'Advanced',font = self.font)
        self.advanced_controls.grid(row = 2,column = 0,sticky = 'ew',pady = (4,0))
        self._build_solve_buttons()
        self._build_param_controls()
        self._build_level_controls()
        self._build_debug_controls()
        self._apply_panel_mode()

    def _build_mode_controls(self):
        """Simple / Advanced の切替を常に見える位置に置く。"""
        self.mode_controls = Tk.Frame(self)
        self.mode_controls.grid(row = 0,column = 0,sticky = 'ew')
        self.mode_label = Tk.Label(self.mode_controls,text = '操作',font = self.font)
        self.mode_label.grid(row = 0,column = 0,sticky = 'w')
        self.mode_button = self._create_button(
            self.mode_controls,
            'Advanced を表示',
            self.toggle_panel_mode,
            row = 0,
            column = 1,
        )
        self.mode_controls.grid_columnconfigure(1,weight = 1)

    def toggle_panel_mode(self):
        """詳細設定の表示・非表示を切り替える。"""
        self.panel_mode = 'advanced' if self.panel_mode == 'simple' else 'simple'
        self._apply_panel_mode()

    def _apply_panel_mode(self):
        if self.panel_mode == 'advanced':
            self.advanced_controls.grid()
            self.mode_button.configure(text = 'Simple に戻る')
        else:
            self.advanced_controls.grid_remove()
            self.mode_button.configure(text = 'Advanced を表示')

    def _build_solve_buttons(self):
        """solve 開始/停止や主要操作ボタンを配置する。"""
        self.reset_button = self._create_button(self.simple_controls,'リセット', self.frame.reset, row = 0, column = 0)
        self.stopper_button = self._create_button(self.simple_controls,'停止', self.frame.stopper, row = 0, column = 1)
        self.my_solve_button = self._create_button(self.simple_controls,'AI に解かせる', self.frame.my_solve, row = 0, column = 2)
        self.open_move_pad_button = self._create_button(self.simple_controls,'手動操作', self.frame.toggle_move_pad, row = 0, column = 3)
        for column in range(4):
            self.simple_controls.grid_columnconfigure(column,weight = 1)

        self.loadparams_all_button = self._create_button(self.advanced_controls,'全AIの設定を読む', self.frame.loadparams_all, row = 0, column = 0)
        self.saveparams_all_button = self._create_button(self.advanced_controls,'全AIの設定を保存', self.frame.saveparams_all, row = 0, column = 1)
        self.tools_button = self._create_button(self.advanced_controls,'ツール', self.frame.open_tools_dialog, row = 0, column = 2)

    def _build_param_controls(self):
        """AI index 指定と param 入出力まわりの入力欄を配置する。"""
        self.param_index_label = self._create_label(self.advanced_controls,'AI 番号', row = 1, column = 0)
        self.param_index_var = Tk.StringVar(value = '0')
        self.param_index_entry = self._create_entry(self.advanced_controls,self.param_index_var, row = 1, column = 1)
        self.loadparams_selected_button = self._create_button(self.advanced_controls,'設定を読む', self.frame.loadparams_selected, row = 1, column = 2)
        self.saveparams_selected_button = self._create_button(self.advanced_controls,'設定を保存', self.frame.saveparams_selected, row = 1, column = 3)
        self.sum_and_var_button = self._create_button(self.advanced_controls,'合計・分散', self.frame.sum_and_var_from_entry, row = 1, column = 8)

    def _build_level_controls(self):
        """level 指定と counter 表示まわりの操作を配置する。"""
        self.level_label = self._create_label(self.advanced_controls,'レベル', row = 1, column = 4)
        self.level_var = Tk.StringVar(value = '0')
        self.level_entry = self._create_entry(self.advanced_controls,self.level_var, row = 1, column = 5)
        self.set_level_button = self._create_button(self.advanced_controls,'レベルを設定', self.frame.set_level_from_entry, row = 1, column = 6)
        self.show_counter_button = self._create_button(self.advanced_controls,'カウンターを表示', self.frame.show_counter_from_entry, row = 1, column = 7)

    def _build_debug_controls(self):
        """viewer/debug 系の入力欄と手動操作ボタンを配置する。"""
        self.grad_index_label = self._create_label(self.advanced_controls,'解析位置', row = 2, column = 0)
        self.grad_index_var = Tk.StringVar(value = str(self.frame.grad_index))
        self.grad_index_entry = self._create_entry(self.advanced_controls,self.grad_index_var, row = 2, column = 1)
        self.grad_mode_label = self._create_label(self.advanced_controls,'解析方法', row = 2, column = 2)
        self.grad_mode_labels = dict(self.GRAD_MODE_CHOICES)
        self.grad_mode_codes = {label:code for code,label in self.GRAD_MODE_CHOICES}
        self.grad_mode_var = Tk.StringVar(value = self.grad_mode_labels[self.frame.grad_mode])
        self.grad_mode_menu = self._create_option_menu(self.advanced_controls,self.grad_mode_var, tuple(self.grad_mode_codes), row = 2, column = 3)
        self.grad_layer_label = self._create_label(self.advanced_controls,'レイヤー', row = 2, column = 4)
        self.grad_layer_var = Tk.StringVar(value = self.frame.grad_layer)
        self.grad_layer_entry = self._create_entry(self.advanced_controls,self.grad_layer_var, row = 2, column = 5)
        self.show_debug_viewer_button = self._create_button(self.advanced_controls,'解析を表示', self.frame.show_debug_viewer_from_entry, row = 2, column = 6, columnspan = 2)

    def selected_grad_mode(self):
        """表示用の名称を既存の解析コード用の内部値へ戻す。"""
        return self.grad_mode_codes[self.grad_mode_var.get()]

    def _create_button(self, master, text, command, row, column, columnspan = 1):
        """共通スタイルの Button を作って grid 配置する。"""
        button = Tk.Button(master,text = text,font = self.font,padx = 1,pady = 1,command = command)
        button.grid(row = row,column = column,columnspan = columnspan,sticky = 'ew')
        return button

    def _create_label(self, master, text, row, column):
        """共通スタイルの Label を作って grid 配置する。"""
        label = Tk.Label(master,text = text,font = self.font)
        label.grid(row = row,column = column,sticky = 'e')
        return label

    def _create_entry(self, master, variable, row, column):
        """共通スタイルの Entry を作って grid 配置する。"""
        entry = Tk.Entry(master,font = self.font,textvariable = variable)
        entry.grid(row = row,column = column,sticky = 'ew')
        return entry

    def _create_option_menu(self, master, variable, values, row, column):
        """共通スタイルの OptionMenu を作って grid 配置する。"""
        option_menu = Tk.OptionMenu(master,variable,*values)
        option_menu.configure(font = self.font)
        option_menu.grid(row = row,column = column,sticky = 'ew')
        return option_menu

    def _configure_columns(self):
        """ControlPanel内の各列を横方向に均等に伸縮させる。"""
        self.grid_columnconfigure(0, weight = 1)
        for column_index in range(9):
            self.advanced_controls.grid_columnconfigure(column_index, weight = 1)
