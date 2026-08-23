"""Top control panel widget."""

import tkinter as Tk

class ControlPanel(Tk.Frame):
    """Frame上部の操作ボタン・入力欄をまとめて配置する。"""

    def __init__(self,master,frame):
        Tk.Frame.__init__(self,master,relief = Tk.RIDGE,bd = 4)
        self.frame = frame
        self.font = ('Century Gothic',12,'bold')
        self._build_buttons()
        self._configure_columns()

    def _build_buttons(self):
        """操作ボタンと入力欄を生成し、grid上に配置する。"""
        self._build_solve_buttons()
        self._build_param_controls()
        self._build_level_controls()
        self._build_debug_controls()

    def _build_solve_buttons(self):
        """solve 開始/停止や主要操作ボタンを配置する。"""
        self.reset_button = self._create_button('Reset', self.frame.reset, row = 0, column = 0)
        self.stopper_button = self._create_button('Stop', self.frame.stopper, row = 0, column = 1)
        self.my_solve_button = self._create_button('start solving', self.frame.my_solve, row = 0, column = 2, columnspan = 2)
        self.loadparams_all_button = self._create_button('loadparams all', self.frame.loadparams_all, row = 0, column = 4)
        self.saveparams_all_button = self._create_button('saveparams all', self.frame.saveparams_all, row = 0, column = 5)
        self.tools_button = self._create_button('tools', self.frame.open_tools_dialog, row = 0, column = 6, columnspan = 2)

    def _build_param_controls(self):
        """AI index 指定と param 入出力まわりの入力欄を配置する。"""
        self.param_index_label = self._create_label('idx', row = 1, column = 0)
        self.param_index_var = Tk.StringVar(value = '0')
        self.param_index_entry = self._create_entry(self.param_index_var, row = 1, column = 1)
        self.loadparams_selected_button = self._create_button('loadparams sel', self.frame.loadparams_selected, row = 1, column = 2)
        self.saveparams_selected_button = self._create_button('saveparams sel', self.frame.saveparams_selected, row = 1, column = 3)
        self.sum_and_var_button = self._create_button('sum&var', self.frame.sum_and_var_from_entry, row = 1, column = 8)

    def _build_level_controls(self):
        """level 指定と counter 表示まわりの操作を配置する。"""
        self.level_label = self._create_label('level', row = 1, column = 4)
        self.level_var = Tk.StringVar(value = '0')
        self.level_entry = self._create_entry(self.level_var, row = 1, column = 5)
        self.set_level_button = self._create_button('set level', self.frame.set_level_from_entry, row = 1, column = 6)
        self.show_counter_button = self._create_button('show counter', self.frame.show_counter_from_entry, row = 1, column = 7)

    def _build_debug_controls(self):
        """viewer/debug 系の入力欄と手動操作ボタンを配置する。"""
        self.grad_index_label = self._create_label('grad idx', row = 2, column = 0)
        self.grad_index_var = Tk.StringVar(value = str(self.frame.grad_index))
        self.grad_index_entry = self._create_entry(self.grad_index_var, row = 2, column = 1)
        self.grad_mode_label = self._create_label('mode', row = 2, column = 2)
        self.grad_mode_var = Tk.StringVar(value = self.frame.grad_mode)
        self.grad_mode_menu = self._create_option_menu(self.grad_mode_var, ('W1','SVD','Grad','IG','Contrast','Occ','PieceOcc','PolicyOcc','PiecePolicyOcc','AttnIn','AttnOut','AttnCentral','EmbNorm','EmbPC1'), row = 2, column = 3)
        self.grad_layer_label = self._create_label('layer', row = 2, column = 4)
        self.grad_layer_var = Tk.StringVar(value = self.frame.grad_layer)
        self.grad_layer_entry = self._create_entry(self.grad_layer_var, row = 2, column = 5)
        self.show_debug_viewer_button = self._create_button('show viewer', self.frame.show_debug_viewer_from_entry, row = 2, column = 6, columnspan = 2)
        self.open_move_pad_button = None

    def _create_button(self, text, command, row, column, columnspan = 1):
        """共通スタイルの Button を作って grid 配置する。"""
        button = Tk.Button(self,text = text,font = self.font,padx = 1,pady = 1,command = command)
        button.grid(row = row,column = column,columnspan = columnspan,sticky = 'ew')
        return button

    def _create_label(self, text, row, column):
        """共通スタイルの Label を作って grid 配置する。"""
        label = Tk.Label(self,text = text,font = self.font)
        label.grid(row = row,column = column,sticky = 'e')
        return label

    def _create_entry(self, variable, row, column):
        """共通スタイルの Entry を作って grid 配置する。"""
        entry = Tk.Entry(self,font = self.font,textvariable = variable)
        entry.grid(row = row,column = column,sticky = 'ew')
        return entry

    def _create_option_menu(self, variable, values, row, column):
        """共通スタイルの OptionMenu を作って grid 配置する。"""
        option_menu = Tk.OptionMenu(self,variable,*values)
        option_menu.configure(font = self.font)
        option_menu.grid(row = row,column = column,sticky = 'ew')
        return option_menu

    def _configure_columns(self):
        """ControlPanel内の各列を横方向に均等に伸縮させる。"""
        for column_index in range(9):
            self.grid_columnconfigure(column_index, weight = 1)
