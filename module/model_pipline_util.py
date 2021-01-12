import pandas as pd
import numpy as np
# Not relevant to the research
#import xgboost as xgb
#import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import pickle as pkl
import time
import warnings
import gc

def oversampling(data, target_column, pos_ratio=.5):
    pos = data[data[target_column] == 1]
    neg = data[data[target_column] == 0]
    n   = data.shape[0]
    if pos[target_column].sum() / n > pos_ratio:
        return data

    repeated_n = int(pos_ratio*n/((1-pos_ratio)*pos.shape[0]))
    data       = pd.concat([pos]*repeated_n+[neg])
    
    return data

def get_obj_type(obj):
    if type(obj).__name__ == 'ABCMeta':
        return obj.__module__.split('.')[0]
    elif type(obj).__name__ == 'module':
        return obj.__name__
    elif type(obj).__name__ == 'type':
        return obj.__module__.split('.')[0]
    else:
        return type(obj).__module__.split('.')[0]
    
def train(model_fn, param, x_train, y_train):
    base_module = get_obj_type(model_fn)
#     if base_module == 'builtins':
#         warnings.warn('deprecated', 'The function will change later.')
        
    if base_module == 'sklearn':
        model = model_fn(**param)
        model.fit(x_train, y_train)
    elif base_module in ['xgboost', 'builtins']:
        data  = xgb.DMatrix(x_train, label=y_train)
        model = xgb.train(param, data)
    elif base_module == 'lightgbm':
        data  = lgb.Dataset(x_train, label=y_train)
        model = lgb.train(param, data)
    else:
        raise Exception('No implementation for {} yet.'.format(base_module))
        
    return model

def predict(model, data, mode):
    base_module = get_obj_type(model)
#     if base_module == 'builtins':
#         warnings.warn('deprecated', DeprecationWarning)
        
    if base_module == 'sklearn':
        if mode == 'classifier':
            y_pred = model.predict_proba(data)
        else:
            y_pred = model.predict(data)
    elif base_module in ['xgboost', 'builtins']:
        data   = xgb.DMatrix(data)
        if mode == 'classifier':
            y_pred = model.predict(data)
        else:
            y_pred = model.predict(data)
    elif base_module == 'lightgbm':
        if mode == 'classfier':
            y_pred = model.predict_proba(data)
        else:
            y_pred = model.predict(data)
    else:
        raise Exception('No implementation for {} yet.'.format(base_module))
        
    return y_pred

def evaluation(model, mode, x_train, y_train, x_val=None, y_val=None, x_test=None, y_test=None):
    matric = {}
    base_module = get_obj_type(model)
    
    # train
    y_score = predict(model, x_train, mode)
    if base_module not in ['xgboost', 'builtins']:
        y_score = y_score[: ,1]
    y_pred  = (y_score > .5).astype(int)
    if mode == 'classifier':
        train_accuracy  = accuracy_score(y_train, y_pred)
        train_precision = precision_score(y_train, y_pred)
        train_recall    = recall_score(y_train, y_pred)
        train_f1        = f1_score(y_train, y_pred)
        train_roc_auc   = roc_auc_score(y_train, y_score)
        
        matric['train_accuracy']  = train_accuracy
        matric['train_precision'] = train_precision
        matric['train_recall']    = train_recall
        matric['train_f1']        = train_f1
        matric['train_roc_auc']   = train_roc_auc
    else:
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        train_mae  = mean_absolute_error(y_train, y_pred)
        train_max  = max_error(y_train, y_pred)
        train_r2   = r2_score(y_train, y_pred)
        
        matric['train_rmse'] = train_rmse
        matric['train_mae']  = train_mae
        matric['train_max']  = train_max
        matric['train_r2']   = train_r2
    
    
    # validation
    if x_val is not None and y_val is not None:
        y_score = predict(model, x_val, mode)
        if base_module not in ['xgboost', 'builtins']:
            y_score = y_score[: ,1]
        y_pred  = (y_score > .5).astype(int)
        if mode == 'classifier':
            val_accuracy  = accuracy_score(y_val, y_pred)
            val_precision = precision_score(y_val, y_pred)
            val_recall    = recall_score(y_val, y_pred)
            val_f1        = f1_score(y_val, y_pred)
            val_roc_auc   = roc_auc_score(y_val, y_score)
            
            matric['val_accuracy']  = val_accuracy
            matric['val_precision'] = val_precision
            matric['val_recall']    = val_recall
            matric['val_f1']        = val_f1
            matric['val_roc_auc']   = val_roc_auc
        else:
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            val_mae  = mean_absolute_error(y_val, y_pred)
            val_max  = max_error(y_val, y_pred)
            val_r2   = r2_score(y_val, y_pred)
            
            matric['val_rmse'] = val_rmse
            matric['val_mae']  = val_mae
            matric['val_max']  = val_max
            matric['val_r2']   = val_r2
    
    # test
    if x_test is not None and y_test is not None:
        y_score = predict(model, x_test, mode)
        if base_module not in ['xgboost', 'builtins']:
            y_score = y_score[: ,1]
        y_pred  = (y_score > .5).astype(int)
        if mode == 'classifier':
            test_accuracy  = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred)
            test_recall    = recall_score(y_test, y_pred)
            test_f1        = f1_score(y_test, y_pred)
            test_roc_auc   = roc_auc_score(y_test, y_score)
            
            matric['test_accuracy']  = test_accuracy
            matric['test_precision'] = test_precision
            matric['test_recall']    = test_recall
            matric['test_f1']        = test_f1
            matric['test_roc_auc']   = test_roc_auc
        else:
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae  = mean_absolute_error(y_test, y_pred)
            test_max  = max_error(y_test, y_pred)
            test_r2   = r2_score(y_test, y_pred)
            
            matric['test_rmse'] = test_rmse
            matric['test_mae']  = test_mae
            matric['test_max']  = test_max
            matric['test_r2']   = test_r2
    
    return matric

def save(model, path, model_name):
    base_module = get_obj_type(model)
    if base_module == 'sklearn':
        with open(path + model_name + '.pkl', 'wb') as f:
            pkl.dump(model, f)
    elif base_module == 'xgboost':
        model.save_model(path + model_name + '.model')
    elif base_module == 'lightgbm':
        model.save_model(path + model_name + '.txt')
    else:
        raise Exception('No implementation for {} yet.'.format(base_module))

def load(path, base_module):
    if base_module == 'sklearn':
        with open(path, 'rb') as f:
            model = pkl.load(f)
    elif base_module == 'xgboost':
        model = xgb.Booster() 
        model.load_model(path)
    elif base_module == 'lightgbm':
        model = lgb.Booster(model_file=path)
    else:
        raise Exception('No implementation for {} yet.'.format(base_module))
        
    return model

def plot_lift_chart(y_pred, y_actual, mode, name, show_last_bin=False):
    df = pd.DataFrame({'y_predict':y_pred, 'y':y_actual})
    df = df.sort_values('y_predict').reset_index(drop=True)

    bins      = 10
    bin_size  = df.shape[0] // bins
    remainder = df.shape[0] % bins

    df['bin'] = df.index // bin_size
    df.loc[df['bin'] == bins, 'bin'] = bins-1
    
    fig, axs = plt.subplots(ncols=2, figsize=(15,5), sharey=True)
    if mode == 'classifier':
        df.groupby('bin')['y'].sum().plot.bar(ax=axs[0])
    else:
        df.groupby('bin')['y'].mean().plot.bar(ax=axs[0])

    boxprops    = dict(linestyle='-', linewidth=2, color='r')
    medianprops = dict(linestyle='-', linewidth=2, color='r')
    bp = df.boxplot(column='y', by='bin', ax=axs[1], boxprops=boxprops,
                    medianprops=medianprops, showfliers=False, showmeans=True, notch=True, return_type='dict')

    [item.set_color('k') for item in bp['y']['boxes']]
    [item.set_color('g') for item in bp['y']['medians']]
    [item.set_color('r') for item in bp['y']['whiskers']]
    [item.set_linestyle('--') for item in bp['y']['whiskers']]
    [item.set_color('k') for item in bp['y']['caps']]

    axs[0].get_shared_x_axes().join(axs[0], axs[1])

    plt.title('')
    fig.suptitle(name)
    plt.tight_layout(pad=4, w_pad=0.5, h_pad=1.0)
    plt.show()
    
    if show_last_bin:
        print('Example of the last bin')
        display(df.loc[df['bin'] == 9, ['y', 'y_predict']].sort_values('y'))

def show_roc_curve(y_test, y_score, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc     = auc(fpr, tpr)

    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC')
    ax.legend(loc="lower right")
    if ax is None:
        plt.show()

def cal_roc_curve(df, target_col, criteria_cols):
    df_temp = df.copy()

    n_pos = df_temp[target_col].sum()
    n_neg = np.abs(df_temp[target_col] - 1).sum()

    df_criteria = df_temp[criteria_cols].drop_duplicates(keep='last')
    df_criteria['true_positive_rate'] = np.nan
    df_criteria['false_positive_rate'] = np.nan

    last_i = -1
    for i, row in df_criteria.iterrows():
        df_criteria.loc[i, 'true_positive_rate'] = df_temp.loc[:i, target_col].sum()/n_pos
        df_criteria.loc[i, 'false_positive_rate'] = np.abs(df_temp.loc[:i, target_col]-1).sum()/n_neg

        last_i = i

    fpr = np.array([0] + df_criteria['false_positive_rate'].tolist())
    tpr = np.array([0] + df_criteria['true_positive_rate'].tolist())

    return fpr, tpr, df_criteria.reset_index(drop=True)[criteria_cols]
    
def show_roc_curve_from_fpr_tpr(fpr, tpr, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    roc_auc     = auc(fpr, tpr)

    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('ROC')
    ax.legend(loc="lower right")
    if ax is None:
        plt.show()
        
def show_confusion_matrix(y_actual, y_pred, labels=[0, 1], title='Confusion matrix', ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    cm = confusion_matrix(y_actual, y_pred)
    cm = cm/np.sum(cm, axis=1)[:,None]

    mat = ax.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    mat.set_clim(0, np.max(cm))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.tick_params(axis='x', which='both', bottom=False, top=False)

    ax.grid(b=False, which='major', axis='both')
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(b=True, which='minor', axis='both', lw=2, color='white')
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:0.2f}'.format(z),c='w' if z > .5 else 'k', ha='center', va='center')

    ax.set_title(title, pad=20)
    plt.colorbar(mat)
    if ax is None:
        plt.show()

def train_and_evaluate_model(name, model_fn, param, mode, save_path, 
                             x_train, y_train, x_val=None, y_val=None,
                             x_test=None, y_test=None):
    start_t = time.time()
    model  = train(model_fn, param, x_train, y_train)
    end_t  = time.time()
    training_time = end_t - start_t
    start_t       = time.time()
    result = evaluation(model, 'classifier', x_train, y_train, x_val, y_val, x_test, y_test)
    end_t  = time.time()
    evaluation_time = end_t - start_t
    result['name'] = name

    start_t     = time.time()
    save(model, save_path, name)
    end_t       = time.time()
    saving_time = end_t - start_t

    result['training_time']   = training_time
    result['evaluation_time'] = evaluation_time
    result['saving_time']     = saving_time
    
    return model, result
        
def run_experiment(models, save_path, mode, x_train, y_train, x_val=None, y_val=None, 
                   x_test=None, y_test=None, label_name=None):
    datasets = ['train']
    if x_val is not None and y_val is not None:
        datasets += ['val']
    if x_test is not None and y_test is not None:
        datasets += ['test']
        
    if mode == 'classifier':
        matrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    else:
        matrics = ['rmse', 'mae', 'max', 'r2']

    columns  = ['name']
    columns += [dataset+'_'+matric for dataset in datasets for matric in matrics]
    columns += ['training_time', 'evaluation_time', 'saving_time']
    results = pd.DataFrame(columns=columns)
    
    for name, item in models.items():
        print(name)
        model_fn, param = item
        model, result = train_and_evaluate_model(name, model_fn, param, 'classifier', save_path, 
                                                 x_train, y_train, x_val, y_val, x_test, y_test)

        df_result = pd.DataFrame({k:[v] for k, v in result.items()})
        df_result[columns].to_csv('{}model_result_{}.csv'.format(save_path, name), index=False)

        results.append(df_result)
        if mode == 'classifier':
            print(' '*10, '\t'.join(datasets))
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            for metric in metrics:
                print(metric.ljust(10, ' '), 
                      '\t'.join(['{:.4f}'.format(df_result.loc[0, '{}_{}'.format(col, metric)]) for col in datasets]))

            print('training time'.ljust(15, ' '), '{:.5f} s'.format(df_result.loc[0, 'training_time']))
            print('evaluation time'.ljust(15, ' '), '{:.5f} s'.format(df_result.loc[0, 'evaluation_time']))
            print('saving time'.ljust(15, ' '), '{:.5f} s'.format(df_result.loc[0, 'saving_time']))
#             df_result.loc[0, '']
        else:
            display(df_result)

        base_module = get_obj_type(model)
        if label_name is None:
            label_name = [0, 1]
        if x_test is not None:
            y_score = predict(model, x_test, mode)
            if base_module not in ['xgboost', 'builtins']:
                y_score = y_score[: ,1]
            y_pred  = (y_score > .5).astype(int)

            if mode == 'classifier':
                fig, axs = plt.subplots(figsize=(15,5), ncols=2)
                show_roc_curve(y_test, y_score, ax=axs[0])
                show_confusion_matrix(y_test, y_pred, label_name, ax=axs[1])
                plt.show()
            plot_lift_chart(y_score, y_test, mode, name)
        elif x_val is not None:
            y_score = predict(model, x_val, mode)
            if base_module not in ['xgboost', 'builtins']:
                y_score = y_score[: ,1]
            y_pred  = (y_score > .5).astype(int)

            if mode == 'classifier':
                fig, axs = plt.subplots(figsize=(15,5), ncols=2)
                show_roc_curve(y_val, y_score, ax=axs[0])
                show_confusion_matrix(y_val, y_pred, label_name, ax=axs[1])
                plt.show()
            plot_lift_chart(y_score, y_val, mode, name)

        del model, y_pred, y_score

        gc.collect()

def show_measurement_matrices(df_matrics, matric_cols, show_validation=False, show_test=False):
    print('Train')
    print(' '*10, ' '.join([v.ljust(18, ' ') for v in df_matrics['name']]))

    for col in matric_cols:
        print(col.ljust(10, ' '), ' '.join(['{:.2f}'.format(v).ljust(18, ' ') for v in df_matrics['train_'+col]]))

    if show_validation:
        print('-'*60)
        print('Validation')
        print(' '*10, ' '.join([v.ljust(18, ' ') for v in df_matrics['name']]))

        for col in matric_cols:
            print(col.ljust(10, ' '), ' '.join(['{:.2f}'.format(v).ljust(18, ' ') for v in df_matrics['val_'+col]]))    

        
    if show_test:
        print('-'*60)
        print('Test')
        print(' '*10, ' '.join([v.ljust(18, ' ') for v in df_matrics['name']]))

        for col in matric_cols:
            print(col.ljust(10, ' '), ' '.join(['{:.2f}'.format(v).ljust(18, ' ') for v in df_matrics['test_'+col]]))  

def plot_roc(df_predictions, y_actual_col='y_actual', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 13))

    for alias in df_predictions.columns[df_predictions.columns != y_actual_col]:
        fpr, tpr, _ = roc_curve(df_predictions[y_actual_col], df_predictions[alias])
        roc_auc     = auc(fpr, tpr)

        lw = 2
        ax.plot(fpr, tpr, lw=lw, label='{} ROC curve (area = {:0.2f})'.format(alias, roc_auc))
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC')
    ax.legend(loc="lower right")

def plot_confusion_matrix(y_actual, y_predict, alias, labels=['0', '1'], ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 13))
        
    cm = confusion_matrix(y_actual, y_predict)
#     cm = cm/np.sum(cm, axis=1)[:,None]
#     cm = cm/df_predictions.shape[0]

    mat = ax.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    mat.set_clim(0, np.max(cm))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.tick_params(axis='x', which='both', bottom=False, top=False)

    ax.grid(b=False, which='major', axis='both')
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(b=True, which='minor', axis='both', lw=2, color='white')
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:,.0f}'.format(z),c='w' if z > y_actual.shape[0]/2 else 'k', ha='center', va='center')

    ax.set_title(alias, pad=5)