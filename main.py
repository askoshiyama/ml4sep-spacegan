import os
import pandas as pd
import spacegan_method
import spacegan_config
import spacegan_utils
import spacegan_selection

if __name__ == "__main__":
    cur_dir = os.getcwd()
    os.chdir(spacegan_config.results_path)

    # neighbours
    df, neighbour_list = spacegan_utils.get_neighbours_featurize(spacegan_config.df, spacegan_config.coord_vars,
                                                             spacegan_config.output_vars, spacegan_config.neighbours)

    # data structures
    target = df[spacegan_config.output_vars].values
    cond_input = df[spacegan_config.cond_vars + neighbour_list].values
    coord_input = df[spacegan_config.coord_vars].values
    spacegan_config.prob_config["output_labels"] = spacegan_config.output_vars
    spacegan_config.prob_config["input_labels"] = spacegan_config.cond_vars + neighbour_list

    # pre-instantiation
    disc_method = spacegan_config.Discriminator(spacegan_config.prob_config["output_dim"],
                                                spacegan_config.prob_config["cond_dim"])
    disc_method.to(spacegan_config.prob_config["device"])
    gen_method = spacegan_config.Generator(spacegan_config.prob_config["cond_dim"],
                                           spacegan_config.prob_config["noise_dim"],
                                           spacegan_config.prob_config["output_dim"])
    gen_method.to(spacegan_config.prob_config["device"])

    # training SpaceGAN
    spacegan = spacegan_method.SpaceGAN(spacegan_config.prob_config, spacegan_config.check_config,
                                        disc_method, gen_method)
    spacegan.train(x_train=cond_input, y_train=target, coords=coord_input)
    spacegan.checkpoint_model(spacegan.epochs) # export final model and data

    # computing metrics
    gan_metrics = spacegan_selection.compute_metrics(target, cond_input, spacegan_config.prob_config,
                                                     spacegan_config.check_config, coord_input,
                                                     spacegan_config.neighbours)

    # selecting and sampling gan
    for criteria in list(spacegan_config.check_config["perf_metrics"].keys()):
        # find best config
        criteria_info = spacegan_config.check_config["pf_metrics_setting"][criteria]
        perf_metrics = gan_metrics[criteria_info["metric_level"]]
        perf_values = criteria_info["agg_function"](perf_metrics[[criteria]])
        best_config = perf_metrics.index[criteria_info["rank_function"](perf_values)]

        # get and set best space gan
        best_spacegan = spacegan_selection.get_spacegan_config(int(best_config), spacegan_config.prob_config,
                                                           spacegan_config.check_config, cond_input, target)
        # training samples
        gan_samples_df = pd.DataFrame(index=range(cond_input.shape[0]),
                                      columns=spacegan_config.cond_vars + neighbour_list + spacegan_config.output_vars)
        gan_samples_df[spacegan_config.cond_vars + neighbour_list] = cond_input
        gan_samples_df[spacegan_config.output_vars] = target
        for i in range(spacegan_config.check_config["n_samples"]):
            gan_samples_df["sample_" + str(i)] = best_spacegan.predict(gan_samples_df[spacegan_config.cond_vars +
                                                                                  neighbour_list])

        # export results
        gan_samples_df.to_pickle("grid_" + criteria + ".pkl.gz")
    spacegan.df_losses.to_pickle("grid_spaceganlosses.pkl.gz")
    gan_metrics["agg_metrics"].to_pickle("grid_checkmetrics.pkl.gz")
