import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.inspection import partial_dependence


def plot_partial_dependence_bootstrap(model, X_train, y_train, features,
                                      feature_name, n_boot=5,
                                      random_state=None):
    rng = check_random_state(random_state)

    # fit a model for each bootstrap sample
    all_estimators = [clone(model) for _ in range(n_boot)]
    for est in all_estimators:
        bootstrap_idx = rng.choice(
            np.arange(X_train.shape[0]), size=X_train.shape[0], replace=True
        )
        X_train_bootstrap = X_train.iloc[bootstrap_idx]
        y_train_bootstrap = y_train[bootstrap_idx]
        est.fit(X_train_bootstrap, y_train_bootstrap)

    # prepare the plotting
    n_fig = 3
    n_rows = (int(len(features) / n_fig) + 1 if len(features) % n_fig != 0
              else int(len(features) / n_fig))
    n_cols = len(features) if n_rows == 1 else n_fig
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(n_cols * 5, n_rows * 5)
    )
    for feat, ax in zip(features, np.ravel(axs)):
        # compute the partial dependence for each models
        X_train_preprocessed = model[0].fit_transform(X_train)
        avg_preds_bootstrap = []
        for est in all_estimators:
            avg_preds, values = partial_dependence(
                est[-1], X_train_preprocessed, feat, grid_resolution=20
            )
            avg_preds_bootstrap.append(avg_preds)

        if len(values) == 2:
            # compute the mean of the average prediction when plotting contour
            # plots
            mean_avg_preds = np.mean(avg_preds_bootstrap, axis=0)
            Z_level = np.linspace(
                mean_avg_preds.min(), mean_avg_preds.max(), 8
            )
            XX, YY = np.meshgrid(values[0], values[1])
            Z = mean_avg_preds[0].T
            CS = ax.contour(XX, YY, Z, levels=Z_level, linewidths=0.5,
                            colors='k')
            ax.contourf(XX, YY, Z, levels=Z_level, vmax=Z_level[-1],
                        vmin=Z_level[0], alpha=0.75)
            ax.clabel(CS, fmt='%2.2f', colors='k', fontsize=10, inline=True)
            ax.set_xlabel(feature_name[feat[0]])
            ax.set_ylabel(feature_name[feat[1]])
        else:
            # plot all average predictions and their mean
            mean_avg_preds = np.zeros_like(avg_preds_bootstrap[0])
            for preds in avg_preds_bootstrap:
                mean_avg_preds += preds
                ax.plot(values[0], preds[0], '--k', linewidth=1, alpha=0.5)
            mean_avg_preds /= len(avg_preds_bootstrap)
            ax.plot(
                values[0], mean_avg_preds[0], 'r', alpha=0.8, label='Average'
            )
            ax.set_xlabel(feature_name[feat])
            ax.set_ylabel('WAGE')
            ax.legend()
    plt.tight_layout()
