# Fitting random forest models #
RF_fit<-function(rec, cv_folds){
  
  model<-parsnip::rand_forest()%>%
    set_mode("regression")%>%
    set_engine("ranger")%>%
    set_args(mtry=tune(),min_n=tune(),trees=tune())
  
  model_wf<-workflow()%>% add_recipe(rec) %>%add_model(model)
  
  ## Parameter grid specification 
  rf_grid <- grid_regular(
    mtry(range = c(30, 100)),
    min_n(range = c(2, 6)),
    trees(range=c(600,1000)),
    levels = 5
  )
  
  doParallel::registerDoParallel()
  model_tuned <- tune::tune_grid(
    object = model_wf,
    resamples = cv_folds, 
    grid = rf_grid,
    metrics = yardstick::metric_set(rmse)
  )
  
  plot<-model_tuned %>%
    collect_metrics() %>%
    filter(.metric=="rmse") %>%
    pivot_longer(cols=c("mtry","min_n","trees"),values_to="value",names_to="parameter")%>%
    ggplot(aes(value, mean, color = parameter))  +
    geom_point(show.legend=FALSE) +
    facet_wrap(~parameter,scales="free_x")
  
  lowest_rmse <- model_tuned %>%
    select_best("rmse", maximize = FALSE) #select the model with the lowest RMSE
  
  final_model <- finalize_workflow(  #Define the best fit model
    model_wf,
    lowest_rmse
  )
  
  final_rf<-finalize_model(model,lowest_rmse)
  
  RMSE_best<-show_best(model_tuned,"rmse",n=1)
  
  vip<-final_rf %>%     #Plotting the variable importance based on the permutation method
    set_engine("ranger", importance = "permutation") %>%
    fit(y ~ .,
        data = juice(prep(rec))
    ) %>%
    vip(geom = "col")
  
  return(list(plot,RMSE_best,vip,final_rf )) 
}


# Fitting MARS models #
Mars<-function(rec, cv_folds){
  
  mars_model<-parsnip::mars()%>%
    set_mode("regression")%>%
    set_engine("earth")%>%
    set_args(num_terms=tune(),prod_degree=tune())
  
  model_wf <- workflow() %>% 
    add_recipe(rec) %>% 
    add_model(mars_model)
  
  ## Parameter grid specification 
  mars_grid <- grid_regular(
    num_terms(range = c(1,30)),
    prod_degree(range = c(1, 3)),
    levels = 10
  )
  
  doParallel::registerDoParallel()
  model_tuned <- tune::tune_grid(
    object = model_wf,
    resamples = cv_folds,
    grid = mars_grid,
    metrics = yardstick::metric_set(rmse)
  )
  
  plot<-model_tuned %>%
    collect_metrics() %>%
    filter(.metric=="rmse") %>%
    pivot_longer(cols=c("num_terms","prod_degree"), values_to="value",names_to="parameter")%>%
    ggplot(aes(value, mean, color = parameter))  +
    geom_point(show.legend=FALSE) +
    facet_wrap(~parameter,scales="free_x")
  
  
  lowest_rmse <- model_tuned %>%
    select_best("rmse", maximize = FALSE) #select the model with the lowest RMSE
  
  final_model <- finalize_workflow(  #Define the best fit model
    model_wf,
    lowest_rmse
  )
  
  final_mars<-finalize_model(mars_model,lowest_rmse)
  
  RMSE_best<-show_best(model_tuned, "rmse", n = 1) #Cross-Validation RMSE for best tuned model
  
  vip<-final_mars %>%      #variable importance plot
    set_engine("earth") %>%
    fit(y ~ .,
        data = juice(prep(rec))
    ) %>%
    vip(geom = "col")
  
  return(list(plot,RMSE_best,vip,final_model)) 
}

# Regularized Regressions #
RegReg <- function(rec, cv_folds){
  
  glmnet_spec <- parsnip::linear_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
    set_engine("glmnet")
  
  lmbda_mixtr_grid <- grid_regular(
    penalty(c(-2,1)), 
    mixture(), 
    levels = 50
  )
  
   wf <- workflow() %>%
     add_recipe(rec) %>% 
     add_model(glmnet_spec)
   
  model_tuned <- tune::tune_grid(
    wf %>% update_model(glmnet_spec),
    resamples = cv_folds,
    grid = lmbda_mixtr_grid,
    metrics = yardstick::metric_set(rmse)
  )
  
  # plot<-model_tuned %>%
  #   collect_metrics() %>%
  #   filter(.metric=="rmse") %>%
  #   pivot_longer(cols=c("num_terms","prod_degree"), values_to="value",names_to="parameter")%>%
  #   ggplot(aes(value, mean, color = parameter))  +
  #   geom_point(show.legend=FALSE) +
  #   facet_wrap(~parameter,scales="free_x")
  # 
  # 
  lowest_rmse <- model_tuned %>%
    select_best("rmse", maximize = FALSE) #select the model with the lowest RMSE
  
  final_model <- finalize_workflow(  #Define the best fit model
    wf %>% update_model(glmnet_spec),
    lowest_rmse
  )
  
  RMSE_best<-show_best(model_tuned, "rmse", n = 1) #Cross-Validation RMSE for best tuned model
  # 
  # vip<-final_model %>% 
  #   fit(y ~ .,
  #       data = juice(prep(rec))
  #   ) %>%
  #   vip(geom = "col")
 
  return(list(RMSE_best,final_model)) 
}

# XgBoost #
XgBoost <- function(rec, cv_folds){

  xgboost_model <- 
    parsnip::boost_tree(
      mode = "regression",
      trees = 1000,
      min_n = tune(),
      tree_depth = tune(),
      learn_rate = tune(),
      loss_reduction = tune()
    ) %>%
    set_engine("xgboost", objective = "reg:squarederror")
  
  # grid specification
  xgboost_params <- 
    dials::parameters(
      min_n(),
      tree_depth(),
      learn_rate(),
      loss_reduction()
    )
  
  xgboost_grid <- 
    dials::grid_max_entropy(
      xgboost_params, 
      size = 100
    )
  
  xgboost_wf <- 
    workflows::workflow() %>%
    add_model(xgboost_model) %>% 
    add_recipe(rec)
  
  
  
  doParallel::registerDoParallel()
  # hyperparameter tuning
  xgboost_tuned <- tune::tune_grid(
    object = xgboost_wf,
    resamples = cv_folds,
    grid = xgboost_grid,
    metrics = yardstick::metric_set(rmse, rsq, mae),
    control = tune::control_grid(verbose = TRUE)
  )
  
  
  
  xgboost_best_params <- xgboost_tuned %>%
    tune::select_best("rmse")

  
  RMSE_best<-show_best(xgboost_tuned, "rmse", n = 1) #Cross-Validation RMSE for best tuned model
  
  
  xgboost_model_final <- xgboost_model %>% 
    finalize_model(xgboost_best_params)
  
  # vip<-final_model %>%      #variable importance plot
  #   set_engine("xgboost") %>%
  #   fit(y ~ .,
  #       data = juice(prep(rec))
  #   ) %>%
  #   vip(geom = "col")
  # 
  return(list(RMSE_best,xgboost_model_final)) 
  
}