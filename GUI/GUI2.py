from utils.import_lib import *
from utils.helper import *
from utils.make_graph import *
from utils.instantiate_net import *

import PySimpleGUI as sg      
sg.theme('DefaultNoMoreNagging')
from shutil import copyfile
from GUI.helper import update_list, show_im, show_im_with_uncertainty, show_im_withdef, draw_figure, print_ROC, print_learning_curves

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasAgg

import tkinter as Tk


# Get names of experiments
all_simu_paths = [root_path, '/data/ADRIC/Simulations_v1']
all_exp = [x for x in os.listdir(root_path + '/Experiments')]
all_exp.sort(key = lambda x : x)

def launch_analysis_box(def_p):
    """ -----------------------------------------------------------------------------------------
    DIALOG BOX
    ----------------------------------------------------------------------------------------- """

    # CURVES ANALYSIS (ROC and training)
    roc_col, roc_row = 260, 160
    nb_COL, nb_ROW   = 5, 50
    n_pred, DS       = 31, 'Mint' # Number of the image for the prediction

    f1_simu = [[sg.Text('Select Location of Simulation Results', text_color='#0f2851', font=("Arial Bold", 11))]]
    f1_simu.append([sg.Combo(values=list(all_simu_paths), key='1_simu', size=(50,5), change_submits=True)])

    f1_exp = [[sg.Text('Select Experiments', text_color='#0f2851', font=("Arial Bold", 11))]]
    f1_exp.append([sg.Text('Query', size=(8, 1)), sg.InputText('', key='1_exp_query', change_submits=True, do_not_clear=True, focus=True)])
    f1_exp.append([sg.Listbox(values=list(all_exp), key='1_exp', size=(130,10), select_mode='multiple', change_submits=True)])
    f1_display = [[sg.Text('Select Items to display (' + str(nb_ROW) +')', size=(30, 1), text_color='#0f2851', font=("Arial Bold", 11))]]

    display      = ['Learning Curve (PSNR)', 'ROC (Clean vs Real corruption)', 'ROC (all)', 'Samples']
    display_path = [ '/Experiments/', '/Results/', '/Results/', '/Results/']
    for idx,x in enumerate(display): 
        f1_display.append([sg.Checkbox(x, key='1_display_'+str(idx))])
    f1_display.append([sg.Button('Intialize',key='1_init', button_color=('white', '#0f2851'))])
    frame1 = [[sg.Column(f1_simu), sg.Column(f1_display), sg.Column(f1_exp)]]

    row = []
    for idx in range(nb_COL):
        this_plot = [[sg.Multiline('', key='1_exp_name'+str(idx), size=(35,2), do_not_clear=True)]]
        for idx2 in range(nb_ROW): 
            this_plot.append([sg.Text('', key='1_txt_' + str(idx2) + '_' + str(idx), size=(35,1))])
            this_plot.append([sg.Canvas(size=(roc_col, roc_row), key='1_im_' + str(idx2) + '_' + str(idx))])
        this_frame = [[sg.Frame('Experiment ' + str(idx+1), this_plot, key='1_frame'+str(idx), title_color='#0f2851', font=("Arial Bold", 11))]]
        row.append(sg.Column(this_frame, scrollable=True, vertical_scroll_only=True) )
    frame1.append(row)

    # GENERAL LAYOUT
    layout = frame1
    layout.append([sg.Button('Cancel', key='Cancel', button_color=('white', '#0f2851'))])
    window = sg.Window('Action selection').Layout(layout).Finalize()
    window.Size = (1850,1000)
    plot_showed = None
    simu_path = all_simu_paths[0]
    new_all_exp = all_exp
    #temp = draw_figure(window.FindElement('1_input').TKCanvas, fig_pred)

    while True:   
        event, values = window.Read()  
        if event is None: 
            break

        # 0- All Analysis : Update default location of simulations
        if '_simu' in event: 
            simu_path = values[event]
            new_all_exp = [x for x in os.listdir(simu_path + '/Experiments')]
            new_all_exp.sort(key = lambda x : x)
            window.FindElement('1_exp').Update(new_all_exp)
            window.FindElement('2_exp').Update(new_all_exp)
            window.FindElement('3_exp').Update(new_all_exp)
                
        # 1- Analysis 1 : Update experiences list
        if event is '1_exp_query':
            new_exp_list = update_list(new_all_exp, values['1_exp_query'].split(','))
            window.FindElement('1_exp').Update(new_exp_list)
            window.FindElement('1_exp_query').Update(values['1_exp_query'])

        # 1- Analysis 1 : Initialize window
        if event is '1_init': 
            disp_idx = 0
            for idx, this_display in enumerate(display): 
                if values['1_display_' + str(idx)] : 
                    for idx2 in range(nb_COL):
                        window.FindElement('1_exp_name'+str(idx2)).Update('')
                        window.FindElement('1_txt_' + str(disp_idx) + '_' + str(idx2)).Update(this_display)
                    disp_idx+= 1
                if disp_idx > nb_ROW-1: 
                    break
            plot_showed = {key: None for key in range(nb_COL)}
            figs        = {key: None for key in range(nb_COL*nb_ROW)}

        # 1- Analysis 1 : Update plot when experience is selected
        if event is '1_exp' and plot_showed is not None: 
            plot_showed_size = 0
            for val in plot_showed: 
                if plot_showed[val] is not None: 
                    plot_showed_size += 1
            is_val_in_dic = [False] * max(plot_showed_size, len(values['1_exp']))
            new_item      = None
            for idx, item in enumerate(values['1_exp']): 
                if item in plot_showed.values(): 
                    is_val_in_dic[idx] = True 
                elif plot_showed_size < nb_COL: 
                    new_item = item

            # Check if item has been removed 
            for idx, key in enumerate(plot_showed.values()):
                if key is not None and key not in values['1_exp']: 
                    plot_showed[idx] = None
                    window.FindElement('1_exp_name'+str(idx)).Update('')

            # Show a new item 
            if new_item: 
                col = nb_COL
                for plot_slot in plot_showed: 
                    if plot_showed[plot_slot] is None: 
                        col = plot_slot
                        break
                if col < nb_COL: 
                    plot_showed[col] = new_item
                window.FindElement('1_exp_name'+str(col)).Update(new_item)
                
                disp_idx = 0
                args = read_json(root_path + '/Experiments/' + new_item + '/00_description.json')
                for j, this_display in enumerate(display):
                    if values['1_display_' + str(j)] and 'Learning' in display[j]:
                        try:
                            fig = print_learningcurves(new_item, GUI=True, row=roc_row, col=roc_col)
                            if figs[col+disp_idx*nb_COL]:
                                figs[col+disp_idx*nb_COL].get_tk_widget().forget()
                            figs[col+disp_idx*nb_COL] = draw_figure(window.FindElement('1_im_' + str(disp_idx) + '_'+str(col)).TKCanvas, fig)
                            disp_idx += 1 
                        except: 
                            pass
                            
                    elif values['1_display_' + str(j)] and 'Real' in display[j]:
                        try: 
                            json_path                 = simu_path + '/Results/' + new_item + '/ROC_clean_vs_realDefaults.json'
                            fig                       = print_ROC(json_path, roc_row, roc_col, AUC=True)
                            if figs[col+disp_idx*nb_COL]:
                                figs[col+disp_idx*nb_COL].get_tk_widget().forget()
                            figs[col+disp_idx*nb_COL] = draw_figure(window.FindElement('1_im_' + str(disp_idx) + '_'+str(col)).TKCanvas, fig)
                        except: 
                            pass
                        disp_idx += 1
                        try: 
                            im_list = [x for x in os.listdir(root_path + '/Results/' + new_item + '/clean_prediction') if '.png' in x]
                            im_list.sort(key = lambda x : int(x.split('_')[0]) if '_' in x  else int(x.split('.')[0]))
                            for i in range(5): 
                                im_path = root_path + '/Results/' + new_item + '/clean_prediction/' + im_list[i]
                                fig = show(read_png(im_path), GUI=True, row=roc_row, col=roc_col)
                                if figs[col+disp_idx*nb_COL]:
                                    figs[col+disp_idx*nb_COL].get_tk_widget().forget()
                                figs[col+disp_idx*nb_COL] = draw_figure(window.FindElement('1_im_' + str(disp_idx) + '_'+str(col)).TKCanvas, fig)
                                disp_idx += 1
                        except: 
                            pass
                            
                    elif values['1_display_' + str(j)] and 'all' in display[j]:
                        try:
                            ROC_list  = [x for x in os.listdir(simu_path + display_path[j] + new_item) if 'json' in x and 'real' not in x]
                        except: 
                            pass
                        for this_json in ROC_list: 
                            try: 
                                json_path                 = simu_path + '/Results/' + new_item + '/' + this_json
                                fig                       = print_ROC(json_path, roc_row, roc_col, AUC=True)
                                if figs[col+disp_idx*nb_COL]:
                                    figs[col+disp_idx*nb_COL].get_tk_widget().forget()
                                figs[col+disp_idx*nb_COL] = draw_figure(window.FindElement('1_im_' + str(disp_idx) + '_'+str(col)).TKCanvas, fig)
                                window.FindElement('1_txt_' + str(disp_idx) + '_' + str(col)).Update(this_json.split('.')[0])
                            except: 
                                pass
                            disp_idx += 1
                            try: 
                                folder_path = (json_path.split('vs_')[1]).split('.')[0] + '_prediction'
                                im_list = [x for x in os.listdir(root_path + '/Results/' + new_item + '/' + folder_path) if '.png' in x]
                                im_list.sort(key = lambda x : int(x.split('_')[0]) if '_' in x  else int(x.split('.')[0]))
                                for i in range(5): 
                                    im_path = root_path + '/Results/' + new_item + '/' + folder_path + '/' + im_list[i]
                                    fig = show(read_png(im_path), GUI=True, row=roc_row, col=roc_col)
                                    if figs[col+disp_idx*nb_COL]:
                                        figs[col+disp_idx*nb_COL].get_tk_widget().forget()
                                    figs[col+disp_idx*nb_COL] = draw_figure(window.FindElement('1_im_' + str(disp_idx) + '_'+str(col)).TKCanvas, fig)     
                                    disp_idx += 1
                            except: 
                                pass
                        
        if event is 'Cancel': 
            values = None
            break
