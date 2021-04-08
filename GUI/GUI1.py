from utils.import_lib import *
from utils.helper import *
from utils.make_graph import *

import PySimpleGUI as sg      
sg.theme('DefaultNoMoreNagging')
from shutil import copyfile
import ast

from utils.launch_code import import_new_db, corrupt_db, get_preview, train_net, evaluate_net
from GUI.helper import update_list, show_im, draw_figure, hjson_parser


# Get path and names of all clean and corrupted datasets
all_clean_ds  = [x for x in os.listdir(data_path)]
all_clean_ds  = [x for x in all_clean_ds if not '.' in x and not x == '__pycache__' and not x == '00_datasets']
all_clean_ds.sort(key = lambda x : x)

all_clean_paths = [data_path + '/' + x for x in all_clean_ds]
all_subfolders_paths = []
all_subfolders      = []
for folder in all_clean_paths : 
    ds_name = folder.split('/')[-1]
    all_subfolders_paths += [folder + '/' + x for x in os.listdir(folder)]
    all_subfolders       += [folder.split('/')[-1] + '_' + x for x in os.listdir(folder)]
nb_subfolder_max = 7

all_ds_name = [data_path + '/' + x for x in os.listdir(data_path)if 'tmp' not in x]
all_ds      = [[this_ds_name.split('/')[-1] + '_' + x for x in os.listdir(this_ds_name + '/Train')] for this_ds_name in all_ds_name]
all_ds      =  [item for sublist in all_ds for item in sublist]
all_ds.sort(key = lambda x : x)

model_classes = [x[len('Class'):-len('.py')] for x in os.listdir(code_path + '/Models') if 'Class' in x and 'Super' not in x]
model_classes.sort(key = lambda x : x)

model_archs = os.listdir(code_path + '/Models/Networks/Architecture_Config')
model_archs.sort(key = lambda x : x)

all_training = os.listdir(code_path + '/Models/Training_Config')
all_training.sort(key = lambda x : x)

# Get names of experiments
all_simu_paths = [root_path, '/data/ADRIC/Simulations_v1']
all_exp = [x for x in os.listdir(all_simu_paths[0] + '/Experiments')]
all_exp.sort(key = lambda x : x)


def launch_dialog_box(def_p):
    """ -----------------------------------------------------------------------------------------
    DIALOG BOX
    ----------------------------------------------------------------------------------------- """
    col  = 20
    col1 = 30
    col2 = 10
    col3 = 5

    # Max number of parameters for data corruption
    nb_param = 0
    for param in def_p:
        if len(def_p[param]) > nb_param : 
            nb_param = len(def_p[param])

    # IMPORT A NEW DATABASE
    frame1 = [[sg.Text('Select which new database you want to import', text_color='#0f2851', font=("Arial Bold", 11))]]
    subframe011 = [[sg.Text('Dataset name', size=(col2, 1)), sg.InputText('', key='0_ds_name', do_not_clear=True, focus=True)]]
    subframe11 = [[sg.Text('Clean Images', size=(col, 1))]]
    subframe11.append([sg.Input(key='0_train_set', do_not_clear=True, focus=True), sg.FolderBrowse()])
    frame1.append([sg.Text('Dataset name', size=(15, 1)), sg.InputText('', key='0_ds_name', do_not_clear=True, focus=True, size=(col, 1)), sg.Text('', size=(18,1)),
        sg.Frame('Training Set', subframe11, title_color='#0f2851', font=("Arial Bold", 11))]) 
    frame1.append([sg.Text('')])
    
    subframe12 = [[sg.Text('Clean Images', size=(col, 1))]]
    subframe12.append([sg.Input(key='0_test_clean', do_not_clear=True, focus=True), sg.FolderBrowse()])
    subframe12.append([sg.Text('Images with Defects', size=(3*col+9, 1)), 
                        sg.Text('Label to identify the defect category', size=(2*col+5, 1)), 
                        sg.Text('Binary mask to segment defects (Optional)', size=(2*col+5, 1)) ])
    for idx in range(9):
        subframe12.append([sg.Text('Type '+str(idx+1), size=(col, 1))])
        subframe12.append([sg.Input(key='0_test_default'+str(idx), change_submits=True, do_not_clear=True, focus=True), sg.FolderBrowse(), 
            sg.Text('', size=(col2, 1)), sg.Input(key='0_test_label'+str(idx), do_not_clear=True, focus=True), 
            sg.Input(key='0_test_mask'+str(idx), change_submits=True, do_not_clear=True, focus=True), sg.FolderBrowse()])
    frame1.append([sg.Frame('Test Set', subframe12, title_color='#0f2851', font=("Arial Bold", 11))]) 

    # CORRUPT AN EXISTING DATABASE
    f2_db = [[sg.Text('Select the Database', text_color='#0f2851', font=("Arial Bold", 11))]]
    for idx,x in enumerate(all_clean_ds): 
        f2_db.append([sg.Checkbox(x, key='1_'+str(idx))])

    f2_default = [[sg.Text('Select the corruption', text_color='#0f2851', font=("Arial Bold", 11))]]
    f2_default.append([sg.Listbox(values= list(def_p.keys()), key='1_default', size=(30,15), change_submits=True, bind_return_key=True)])
    f2_default.append([sg.Button('Set default parameters',key='1_def_param', button_color=('white', '#0f2851'))])
    f2_default.append([sg.Checkbox('Histogram Equalization', key='1_HE')])

    f2_param = [[sg.Text('Default Parameters', size=(col, 1), key='def_name')]]
    f2_param.append([sg.Text('Corruption Name', size=(col, 1)), sg.InputText('', key='1_cor_name', do_not_clear=True, focus=True)])
    for i in range(nb_param): 
        f2_param.append([sg.Text('Param' + str(i), size=(col2, 1), key = '1_pp_'+str(i)), sg.InputText('', key='1_p_'+str(i), do_not_clear=True, focus=True)])
    f2_param.append([sg.Button('Preview',key='1_preview', button_color=('white', '#0f2851'), change_submits=True)])
    f2_param.append([sg.Canvas(size=(300, 200), key='canvas')])
    frame2 = [[sg.Column(f2_db), sg.Column(f2_default), sg.Column(f2_param)]]

    # TRAIN A NETWORK
    # Select the dataset
    f3_ds01 = []
    f3_ds01 = [[sg.Text('Query', size=(8, 1)), sg.InputText('', key='3_ds_query', change_submits=True, do_not_clear=True, focus=True)]]
    f3_ds01.append([sg.Listbox(values=all_ds, key='3_ds', size=(40,15), select_mode='multiple', bind_return_key=True)])
    f3_ds01.append([sg.Text('')])
    f3_ds01.append([sg.Text('Resolution (1st axis)', size=(23, 1)), sg.InputText('', key='3_dimension', size=(10, 1), change_submits=True, do_not_clear=True, focus=True)])
    f3_ds = [[sg.Frame('Select the Dataset(s)', f3_ds01, title_color='#0f2851', font=("Arial Bold", 11))]]

    # Select Model Class and Architecture
    f3_model01 = [[sg.Text('Select the Model Class (choose ImToIm by default)', text_color='#0f2851', font=("Arial Bold", 11))]]
    f3_model01.append([sg.Combo(values=list(model_classes), key='3_model_class', size=(50,5), change_submits=True)])
    f3_model01.append([sg.Text('Select the Model Architecture', text_color='#0f2851', font=("Arial Bold", 11))])
    f3_model01.append([sg.Text('Query', size=(8, 1)), sg.InputText('', key='3_model_query', change_submits=True, do_not_clear=True, focus=True)])
    f3_model01.append([sg.Listbox(values=list(model_archs), key='3_model_arch', size=(80,10), change_submits=True,  select_mode='multiple', bind_return_key=True)])
    f3_model01.append([sg.Text('Save this new Model Architecture under the name:')])
    f3_model01.append([sg.InputText('', key='3_archi_name', size=(40,10), do_not_clear=True, focus=True), sg.Button('Save',key='3_save_archi', button_color=('white', '#0f2851'))])
    f3_model01.append([sg.Multiline('', key='3_archi_hjson', size=(50,20), do_not_clear=True, focus=True)])
    
    f3_model = [[sg.Frame('Select the Model Class and Architecture', f3_model01, title_color='#0f2851', font=("Arial Bold", 11))]]

    # Select Training Scheme
    f3_train01 = [[sg.Text('Query', size=(8, 1)), sg.InputText('', key='3_train_query', change_submits=True, do_not_clear=True, focus=True)]]
    f3_train01.append([sg.Listbox(values=all_training, key='3_train', size=(60,10), change_submits=True, select_mode='multiple', bind_return_key=True)])
    f3_train01.append([sg.Text('Save this new Training Scheme under the name:')])
    f3_train01.append([sg.InputText('', key='3_train_name', size=(40,10), do_not_clear=True, focus=True), sg.Button('Save',key='3_save_train', button_color=('white', '#0f2851'))])
    f3_train01.append([sg.Multiline('', key='3_train_hjson', size=(50,20), do_not_clear=True, focus=True)])

    f3_train = [[sg.Frame('Select the Model Training Scheme', f3_train01, title_color='#0f2851', font=("Arial Bold", 11))]]

    frame3 = [[sg.Column(f3_ds), sg.Column(f3_model), sg.Column(f3_train)]]

    # EVALUATE A NETWORK
    f4_simu = [[sg.Text('Select Location of Simulation Results', text_color='#0f2851', font=("Arial Bold", 11))]]
    f4_simu.append([sg.Combo(values=list(all_simu_paths), key='4_simu', size=(50,5), change_submits=True)])

    f4_exp = [[sg.Text('Select Experiments', text_color='#0f2851', font=("Arial Bold", 11))]]
    f4_exp.append([sg.Text('Query', size=(8, 1)), sg.InputText('', key='4_exp_query', change_submits=True, do_not_clear=True, focus=True)])
    f4_exp.append([sg.Listbox(values=list(all_exp), key='4_exp', size=(120,10), select_mode='multiple', change_submits=True, bind_return_key=True)])
    frame4 = [[sg.Column(f4_simu), sg.Column(f4_exp)]]

    sub_frame41 = [[sg.Text('This options reprint ROC curves of an experiment given that TP and FP rates have been computed previously.')]]
    sub_frame41.append([sg.Checkbox('Standard evaluation (Image-wise)', key='4_ROC')])
    sub_frame41.append([sg.Checkbox('Standard evaluation (Pixel-wise)', key='4_ROC_pixel')])
    sub_frame41.append([sg.Text('_________________________________________________________')])
    sub_frame41.append([sg.Checkbox('Print Prediction', key='4_ROC_printpred')])
    frame4.append([sg.Frame('Standard ROC Curves', sub_frame41, title_color='#0f2851', font=("Arial Bold", 11))])

    sub_frame42 = [[sg.Text('MCDropout allows to evaluate uncertainty of a network by introducting dropout during inference. By averaging results obtained with multiple dropout patterns, uncertainty is estimated.')]]
    sub_frame42.append([sg.Checkbox('Perform MCdropout evaluation (Image-wise)', key='4_MCdrop')])
    sub_frame42.append([sg.Checkbox('Perform MCdropout evaluation (Pixel-wise)', key='4_MCdrop_pixel')])
    sub_frame42.append([sg.Text('_________________________________________________________')])
    sub_frame42.append([sg.Checkbox('Print Prediction', key='4_MCdrop_printpred')])
    frame4.append([sg.Frame('Monte Carlo Dropout', sub_frame42, title_color='#0f2851', font=("Arial Bold", 11))])


    # GENERAL LAYOUT
    layout = [[sg.TabGroup([[sg.Tab('IMPORT_DB', frame1), sg.Tab('CORRUPT_DB', frame2), sg.Tab('TRAIN_NET', frame3),  sg.Tab('EVAL_NET', frame4)]], key='action_choice')]]    
    layout.append([sg.Button('Execute',key='Execute', button_color=('white', '#0f2851')), sg.Button('Cancel', key='Cancel', button_color=('white', '#0f2851')), 
                sg.Button('Add to the TODO list', key='TODO', button_color=('white', '#34916f'))])
    window = sg.Window('Action selection').Layout(layout)   
    new_all_exp = all_exp
    simu_path = all_simu_paths[0]
    while True:   
        event, values = window.Read()  
        if event is None: 
            break

        # 1- Import a new database : Update labels
        if '0_test_default' in event: 
            idx = event[-1]
            window.FindElement('0_test_label'+str(idx)).Update(values['0_test_default'+str(idx)].split('/')[-1]) 

        # 2- Corrupt ds : Update parameters list 
        if event is '1_def_param':
            try: 
                params  = def_p[values['1_default'][0]]
                counter = 0
                for key, value in params.items(): 
                    window.FindElement('1_pp_'+str(counter)).Update(key)  
                    window.FindElement('1_p_'+str(counter)).Update(value)  
                    counter += 1
                for i in range(counter, nb_param):
                    window.FindElement('1_pp_'+str(i)).Update('')  
                    window.FindElement('1_p_'+str(i)).Update('XXX')
            except: 
                print('unable to determine default type')

        # 2- Corrupt ds : Update preview 
        if event is '1_preview': 
            # Keep parameters modifications 
            counter = 0
            for key, value in params.items(): 
                window.FindElement('1_pp_'+str(counter)).Update(key)  
                window.FindElement('1_p_'+str(counter)).Update(values['1_p_'+str(counter)])  
                counter += 1
            for i in range(counter, nb_param):
                window.FindElement('1_pp_'+str(i)).Update('')  
                window.FindElement('1_p_'+str(i)).Update('XXX')
            # Provide the preview
            img = get_preview(values)
            fig = show_im(img[0], 200, 300)
            _ = draw_figure(window.FindElement('canvas').TKCanvas, fig)

        # 3- Create ds : Update subfolders list 
        if event is '2_subfolders': 
            new_clean_ds = values['2_all_clean_ds']
            new_clean_paths = [data_path + '/' + x for x in new_clean_ds]
            new_subfolders_paths = []
            new_subfolders      = []
            for folder in new_clean_paths : 
                new_subfolders_paths += [folder + '/' + x for x in os.listdir(folder)]
                new_subfolders       += [folder.split('/')[-1] + '_' + x for x in os.listdir(folder)]

            window.FindElement('2_all_subfolders').Update(new_subfolders)
            window.FindElement('2_training_prop').Update(values['2_training_prop'])
            window.FindElement('2_val_prop').Update(values['2_val_prop'])

        # 3- Create ds : Update subfolders list for training and eval repartition
        if event is '2_split': 
            ds_to_split = values['2_all_subfolders']
            for idx,ds in enumerate(ds_to_split): 
                window.FindElement('2_training_' + str(idx)).Update(ds)
            if idx > nb_subfolder_max: 
                break

        # 4- Train net : Update datasetlist
        if event is '3_ds_query':
            new_list = update_list(all_ds, values['3_ds_query'].split(','))
            window.FindElement('3_ds').Update(new_list)
            window.FindElement('3_ds_query').Update(values['3_ds_query'])

        # 4- Train net : Update modellist
        if event is '3_model_query':
            model_archs_bis = os.listdir(code_path + '/Models/Networks/Architecture_Config')
            model_archs_bis.sort(key = lambda x : x)
            new_list = update_list(model_archs_bis, values['3_model_query'].split(','))
            window.FindElement('3_model_arch').Update(new_list)
            window.FindElement('3_model_query').Update(values['3_model_query'])

        # 4- Train net : Update trainlist
        if event is '3_train_query':
            all_training_bis = os.listdir(code_path + '/Models/Training_Config')
            all_training_bis.sort(key = lambda x : x)
            new_list = update_list(all_training_bis, values['3_train_query'].split(','))
            window.FindElement('3_train').Update(new_list)
            window.FindElement('3_train_query').Update(values['3_train_query'])

        # 4- Train net : Update trainingHJSON
        if event is '3_train' and values['3_train'] != []:
            message = hjson_parser(code_path + '/Models/Training_Config/' + values['3_train'][0])
            window.FindElement('3_train_hjson').Update(message)

        # 4- Train net : Save new training config
        if event is '3_save_train': 
            if values['3_train_name'] == '': 
                sg.Popup('Your configuration name is empty')
            
            all_configs = os.listdir(code_path + '/Models/Training_Config')
            if values['3_train_name']+ '.hjson' in all_configs: 
                sg.Popup('One training configuration file already has this name')
            else: 
                with open(code_path + '/Models/Training_Config/' + values['3_train_name'] + '.hjson', 'w') as fp:
                    hjson.dump(hjson.loads(values['3_train_hjson']), fp, sort_keys=True, indent=4)
                    model_train_bis = os.listdir(code_path + '/Models/Training_Config')
                    model_train_bis.sort(key = lambda x : x)
                    new_list = update_list(model_train_bis, values['3_train_query'].split(','))
                    window.FindElement('3_train').Update(new_list)

        # 4- Train net : Update architectureHJSON
        if event is '3_model_arch' and values['3_model_arch'] != []:
            message = hjson_parser(code_path + '/Models/Networks/Architecture_Config/' + values['3_model_arch'][0])
            window.FindElement('3_archi_hjson').Update(message)

        # 4- Train net : Save new architecture config 
        if event is '3_save_archi': 
            if values['3_archi_name'] == '': 
                sg.Popup('Your configuration name is empty')
            
            all_configs = os.listdir(code_path + '/Models/Networks/Architecture_Config')
            if values['3_archi_name']+ '.hjson' in all_configs: 
                sg.Popup('One architecture configuration file already has this name')
            else: 
                with open(code_path + '/Models/Networks/Architecture_Config/' + values['3_archi_name'] + '.hjson', 'w') as fp:
                    hjson.dump(hjson.loads(values['3_archi_hjson']), fp, sort_keys=True, indent=4)
                    model_archs_bis = os.listdir(code_path + '/Models/Networks/Architecture_Config')
                    model_archs_bis.sort(key = lambda x : x)
                    new_list = update_list(model_archs_bis, values['3_model_query'].split(','))
                    window.FindElement('3_model_arch').Update(new_list)

        # 5- Evaluate net : Update default location of simulations
        if '4_simu' in event: 
            simu_path = values[event]
            new_all_exp = [x for x in os.listdir(simu_path + '/Experiments')]
            new_all_exp.sort(key = lambda x : x)
            window.FindElement('4_exp').Update(new_all_exp)

        #5 - Evaluate net : Update experiences list
        if event is '4_exp_query':
            new_exp_list = update_list(new_all_exp, values['4_exp_query'].split(','))
            window.FindElement('4_exp').Update(new_exp_list)
            window.FindElement('4_exp_query').Update(values['4_exp_query'])

        if event is 'Execute': 
            break

        if event is 'TODO': 
            action    = values['action_choice']
            if action == 'CREATE_DS':
                print('The creation of a datase must be executed directly : click on execute')
            else: 
                todo_path = code_path + '/Todo_list'
                if not os.path.exists(todo_path):
                    os.makedirs(todo_path)
                    lst = None
                else: 
                    lst = os.listdir(todo_path)

                counter = 0
                for file in lst: 
                    if action in file: 
                        counter += 1 
                write_json(values, code_path + '/Todo_list/' + action + '_' + str(counter) + '.json')

        if event is 'Cancel': 
            values = None
            break


    """ -----------------------------------------------------------------------------------------
    EXECUTE ACTION DEPENDING ON THE OUTPUT OF THE DIALOG BOX
    ----------------------------------------------------------------------------------------- """
    # Execute action choosen before Execute
    if values is not None: 
        if values['action_choice'] == 'IMPORT_DB': 
            import_new_db(values)
        elif values['action_choice'] == 'CORRUPT_DB': 
            corrupt_db(values, def_p)
        elif values['action_choice'] == 'CREATE_DS': 
            create_ds(values, new_subfolders_paths)
        elif values['action_choice'] == 'TRAIN_NET': 
            train_net(values)
        elif values['action_choice'] == 'EVAL_NET': 
            evaluate_net(values)

