import keras.backend as K
from keras.models import Model

#https://github.com/philipperemy/keract

def _evaluate(model: Model, nodes_to_evaluate, x, y=None):
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, nodes_to_evaluate)
    x_, y_, sample_weight_ = model._standardize_user_data(x, y)
    return f(x_ + y_ + sample_weight_)


def get_gradients_of_trainable_weights(model, x, y):
    nodes = model.trainable_weights
    nodes_names = [w.name for w in nodes]
    return _get_gradients(model, x, y, nodes, nodes_names)


def get_gradients_of_activations(model, x, y, layer_name=None):
    nodes = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]
    nodes_names = [n.name for n in nodes]
    return _get_gradients(model, x, y, nodes, nodes_names)


def _get_gradients(model, x, y, nodes, nodes_names):
    if model.optimizer is None:
        raise Exception('Please compile the model first. The loss function is required to compute the gradients.')
    grads = model.optimizer.get_gradients(model.total_loss, nodes)
    gradients_values = _evaluate(model, grads, x, y)
    result = dict(zip(nodes_names, gradients_values))
    return result


def get_activations(model, x, layer_name=None):
    nodes = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None]
    # we process the placeholders later (Inputs node in Keras). Because there's a bug in Tensorflow.
    input_layer_outputs, layer_outputs = [], []
    [input_layer_outputs.append(node) if 'input_' in node.name else layer_outputs.append(node) for node in nodes]
    activations = _evaluate(model, layer_outputs, x, y=None)
    activations_dict = dict(zip([output.name for output in layer_outputs], activations))
    activations_inputs_dict = dict(zip([output.name for output in input_layer_outputs], x))
    result = activations_inputs_dict.copy()
    result.update(activations_dict)
    return result


def display_activations(activations_dict, file_path):
    import numpy as np
    import matplotlib.pyplot as plt
    images_per_row = 32
    f= plt.figure(figsize=(20,35))

    def stack_ims(vec): 
        this_row = []
        for idx,im in enumerate(temp): 
            if idx == 0: 
                this_row = im 
            else: 
                this_row = np.hstack((this_row, im))
        return this_row

    nb_feat_maps_to_show = 0
    for name in sorted(activations_dict.keys()):
        activation_map = activations_dict[name]
        if len(activation_map.shape) == 4 and '-layer_'in name: 
            nb_feat_maps_to_show += 1

    fig_num = 1
    for name in sorted(activations_dict.keys()):
        activation_map = activations_dict[name]
        if len(activation_map.shape) == 4 and '-layer_'in name: 
            #print('Displaying activation map [{}]'.format(name))
            shape = activation_map.shape
            im_row, im_col, nb_feat = shape[1], shape[2], shape[3]
            nb_ROW = nb_feat // images_per_row
            nb_in_last_row = nb_feat % images_per_row

            this_feat_map = []
            for this_row in range(nb_ROW+1): 
                if this_row == 0: 
                    temp = np.transpose(np.mean(activation_map, axis=0), (2, 0, 1)) 
                    #temp = np.transpose(activation_map[0], (2, 0, 1)) - np.transpose(np.mean(activation_map, axis=0), (2, 0, 1)) 
                    temp = temp[this_row*images_per_row : (this_row+1)*images_per_row]
                    this_feat_map = stack_ims(temp)
                elif this_row != nb_ROW: 
                    temp = np.transpose(np.mean(activation_map, axis=0), (2, 0, 1)) 
                    #temp = np.transpose(activation_map[0], (2, 0, 1)) - np.transpose(np.mean(activation_map, axis=0), (2, 0, 1)) 
                    temp = temp[this_row*images_per_row : (this_row+1)*images_per_row]
                    this_feat_map = np.vstack((this_feat_map, stack_ims(temp))) 
                # plot only complete rows...

            fig = f.add_subplot(nb_feat_maps_to_show, 1, fig_num)
            fig.set_title(name, fontsize=12)
            im = plt.imshow(this_feat_map, cmap='jet')
            f.colorbar(im, orientation='vertical')
            fig.axis('off')
        fig_num += 1

    print('save result', file_path)
    plt.savefig(file_path)
    #plt.savefig('test.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()




