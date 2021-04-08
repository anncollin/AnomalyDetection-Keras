from utils.import_lib import *
from utils.launch_code import import_new_db, corrupt_db, train_net, evaluate_net

""" COLLIN Anne-Sophie """ 

""" -----------------------------------------------------------------------------------------
MAIN : launch GUI or execute action described by the todo list 
----------------------------------------------------------------------------------------- """
parser = argparse.ArgumentParser()
# By default : execute to do list or Launch GUI1 
parser.add_argument('-gui1', action='store_true')
# By default : execute to do list or Launch GUI2
parser.add_argument('-gui2', action='store_true')
# By default : delete dictionnary from todo list when executed or don't delete it
parser.add_argument('-keep', action='store_true')
# Name of anoter folder containing other dictionnaries to execute code
parser.add_argument('--todo', type=str)

def main(args):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(code_path + '/Todo_list'):
        os.makedirs(code_path + '/Todo_list')

    if args.gui1:
        from GUI.GUI1 import launch_dialog_box
        launch_dialog_box(all_p[0])
    elif args.gui2: 
        from GUI.GUI2 import launch_analysis_box
        launch_analysis_box(all_p[0])
    else: 
        if args.todo: 
            todo_path = code_path + '/' + args.todo
        else:
            todo_path = code_path + '/Todo_list'

        lst = os.listdir(todo_path)
        lst.sort(key = lambda x : x)
        for file in lst:  
            if 'IMPORT_DB' in file: 
                import_new_db(read_json(todo_path + '/' + file))
            elif 'CORRUPT_DB' in file: 
                corrupt_db(read_json(todo_path + '/' + file), all_p[0])
            elif 'TRAIN_NET' in file: 
                train_net(read_json(todo_path + '/' + file))
            elif 'EVAL_NET' in file: 
                evaluate_net(read_json(todo_path + '/' + file))
            if not args.keep: 
                os.remove(todo_path + '/' + file)


args = None
if __name__ == '__main__':
    args = parser.parse_args()
main(args)
