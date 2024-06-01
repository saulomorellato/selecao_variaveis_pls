## PACOTES ##

library(MASS)       # stepwise (aic)
library(glmulti)    # all regression
library(car)        # multicolinearidade
library(tidyverse)  # manipulacao de dados
library(tidymodels) # modelos de machine learning
library(plsmod)     # uso de pls no tidymodels
library(glmtoolbox) # testes para regressao logistica
library(ecostats)   # grafico de envelope do modelo
library(gtsummary)  # visualizacao dos resultados da regressao
library(janitor)    # limpeza de dados
library(tictoc)     # cronometrar tempo de execucao
library(vip)        # extrair a importancia de cada variavel
library(DALEX)      # extrair a importancia de cada variavel via permutacao
library(DALEXtra)   # complemento do pacote DALEX





##### CARREGANDO OS DADOS #####

df<- readxl::read_xlsx("dadosVinicius.xlsx") %>% data.frame()



##### MANIPULANDO OS DADOS #####

df<- df %>% clean_names()
names(df)<- gsub("_1", "", names(df))

#df %>% glimpse()  # breve visualizacao

#df<- na.omit(df)
df<- df %>% filter(!is.na(grave))
df<- df %>% dplyr::select(-id)
df$grave<- df$grave %>% factor()



##### SPLIT #####

set.seed(0)
split<- initial_split(df, strata=grave, prop=0.8)

df.train<- training(split)
df.test<- testing(split)

folds<- vfold_cv(df, v=2, repeats=5, strata=grave)



##### PRÉ-PROCESSAMENTO #####

receita<- recipe(grave ~ . , data = df) %>%
  #step_zv(all_predictors()) %>% 
  step_nzv(all_predictors(),freq_cut=tune()) %>% 
  step_filter_missing(all_predictors(),threshold = 0.4) %>% 
  #step_normalize(all_numeric_predictors()) %>% 
  step_impute_knn(all_predictors()) %>%
  #step_naomit() %>% 
  step_corr(all_numeric_predictors(),threshold=tune())







##### MODELOS #####

model_pls<- parsnip::pls(num_comp = tune(),
                         predictor_prop = tune()) %>%
  set_engine("mixOmics") %>%
  set_mode("classification")




##### WORKFLOW #####

wf_pls<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_pls)




##### HIPERPARAMETERS TUNING - BAYESIAN SEARCH #####

tic()
tune_pls<- tune_bayes(wf_pls,
                      resamples = folds,
                      initial = 5,
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=FALSE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(num_comp(range(1, 20)),
                                              predictor_prop(range=c(0,1)),
                                              threshold(range=c(0.8,1)),
                                              freq_cut(range=c(5,50)))
)
toc()
# 91.91 sec elapsed



## VISUALIZANDO OS MELHORES MODELOS (BEST RMSE)

show_best(tune_pls,n=3)




## MODELOS TREINADOS APOS TUNAR OS HIPERPARAMETROS

wf_pls_trained<- wf_pls %>% finalize_workflow(select_best(tune_pls)) %>% fit(df)




## SALVANDO OS MODELOS

saveRDS(wf_pls_trained,"wf_pls_trained.rds")




## CARREGANDO OS MODELOS SALVOS

# wf_pls_trained<- readRDS("wf_pls_trained.rds")




## COEFICIENTES DOS MODELOS TREINADOS

k<- 30        # numero de variaveis selecionadas

variaveis_selecionadas<- vip::vi(wf_pls_trained) %>% dplyr::select(Variable)
variaveis_selecionadas<- variaveis_selecionadas$Variable[1:k]

variaveis_selecionadas


pfun<- function(object, newdata){
  p<- predict(object, new_data=newdata, type="prob") %>% data.frame()
  #p$estimate<- p$estimate %>% as.vector()
  p<- p[,2] %>% as.vector()
  p
}

tic()
vip::vi_permute(wf_pls_trained,
                train=df,
                target="grave",
                metric="roc_auc",
                pred_wrapper=pfun,
                event_level="second", 
                nsim=1)
toc()
# 



explainer<- explain_tidymodels(model=wf_pls_trained,
                               data=dplyr::select(df,-grave),
                               y=df$grave==1,
                               label="Partial Least Squares")

tic()
vi_pls<- model_parts(explainer,
                     type="variable_importance")
toc()
# 


tic()
vi_pls_perm<- vi_pls %>% as.data.frame() %>% 
  dplyr::filter(variable!="_full_model_",
                variable!="_baseline_") %>% 
  dplyr::rename(importance = dropout_loss) %>% 
  dplyr::select(-c(permutation,label)) %>% 
  group_by(variable) %>% 
  summarise(importance = mean(importance)) %>% 
  dplyr::arrange(desc(importance))
toc()








receita_pls<- recipe(grave ~ . , data = df) %>%
  step_zv(all_predictors()) %>% 
  step_filter_missing(all_predictors(),threshold = 0.4) %>% 
  step_naomit() %>% 
  step_pls(all_numeric_predictors(),
           outcome="grave",
           num_comp=as.numeric(show_best(tune_pls,n=1)[1]),
           predictor_prop=as.numeric(show_best(tune_pls,n=1)[2]))
  

library(GGally)
receita_pls %>% prep() %>% bake(new_data=df) %>% 
  ggpairs(columns = 2:3, ggplot2::aes(colour=grave))


receita_pls %>% prep() %>% bake(new_data=df) %>% tidy()


plot_components<- function(receita, dados=df){
  receita %>% 
    prep() %>% 
    bake(new_data=dados) %>% 
    ggplot(aes(x=.panel_x, y=.panel_y, color=grave, fill=grave)) +
    geom_point(alpha=0.4, size=0.5) +
    #geom_autodensity(alpha=0.3) +
    geom_density(alpha=0.3) +
    facet_matrix(vars(-grave), layer.diag=2) +
    scale_color_brewer(palette="Dark2") +
    scale_fill_brewer(palette="Dark2")
}

receita_pls %>% plot_components()
  
  





##### DADOS COM AS VARIAVEIS SELECIONADAS #####

df2<- df %>% dplyr::select(all_of(c("grave",variaveis_selecionadas)))




##### VERIFICANDO MULTICOLINEARIDADE #####

# remover ou transformar a variavel com valor maior que 10

var_inflat_factor<- glm(grave ~ ., family=binomial, data=df2) %>% vif()
while(max(var_inflat_factor)>10){
  df2<- df2 %>% dplyr::select(-names(df2)[order(-var_inflat_factor)[1]+1])
  var_inflat_factor<- glm(grave ~ ., family=binomial, data=df2) %>% vif()
}

glm(grave ~ ., family=binomial, data=df2) %>% vif()




### PREPARANDO MODELO DE REGRESSAO ###

modelo_base<- glm(grave ~ ., family=binomial, data=df2)
modelo_base %>% summary()



### SELECAO DE VARIAVEIS - STEPWISE (AIC) ###

modelo_step<- stepAIC(modelo_base, direction="both")
modelo_step %>% summary()
modelo_step %>% tbl_regression(exponentiate = TRUE,
                               estimate_fun = function(x) style_ratio(x, digits = 4))




##### TODOS MODELOS DE REGRESSAO #####

tic()
modelo_all_reg<- glmulti(grave ~ .,
                         data = df2,
                         crit = aic,         # aic, aicc, bic, bicc
                         level = 1,          # 1 sem interacoes, 2 com
                         method = "g",       # "d", ou "h", ou "g"
                         family = binomial,
                         fitfunction = glm,  # tipo de modelo (lm, glm, etc)
                         report = FALSE,
                         plotty = FALSE
)
toc()

modelo_all_reg<- modelo_all_reg@objects[[1]]
modelo_all_reg %>% summary()
modelo_all_reg %>% tbl_regression(exponentiate = TRUE,
                                  estimate_fun = function(x) style_ratio(x, digits = 4))





### TESTE HOSMER-LEMESHOW DE QUALIDADE DO AJUSTE ###

hltest(modelo_step,verbose=FALSE)$p.value

# é um teste qui-quadrado de aderência
# H0: não existe diferença entre observados e esperados
# H1: há diferença
# desejamos um valor MAIOR que 0.05




### GRÁFICOS DO MODELO ###

n<- nrow(df2)    		                # número de observações
k<- length(modelo_step$coef)        # k=p+1 (número de coeficientes)

corte.hii<- 2*k/n		                # corte para elementos da diagonal de H
corte.cook<- qf(0.5,k,n-k)      	  # corte para Distância de Cook

hii<- hatvalues(modelo_step) 		    # valores da diagonal da matriz H
dcook<- cooks.distance(modelo_step)	# distância de Cook

obs<- 1:n

df.fit<- data.frame(obs,hii,dcook)


# GRÁFICO - ALAVANCAGEM

# df.fit %>% ggplot(aes(x=obs,y=hii,ymin=0,ymax=hii)) + 
#   geom_point() + 
#   geom_linerange() + 
#   geom_hline(yintercept = corte.hii, color="red", linetype="dashed") + 
#   xlab("Observação") + 
#   ylab("Alavancagem") + 
#   theme_bw()



# GRÁFICO - DISTÂNCIA DE COOK - PONTOS INFLUENTES

df.fit %>% ggplot(aes(x=obs,y=dcook,ymin=0,ymax=dcook)) + 
  geom_point() + 
  geom_linerange() +
  geom_hline(yintercept = corte.cook, color="red", linetype="dashed") + 
  xlab("Observação") + 
  ylab("Distância de Cook") + 
  theme_bw()



# ENVELOPE

env<- plotenvelope(modelo_step,
                   which=2,
                   n.sim=10000,
                   conf.level=0.95,
                   plot.it=FALSE) 

env[[2]]$p.value    # H0: modelo correto vs. H1: modelo incorreto (desejamos um valor MAIOR que 0.05)

df.env<- data.frame(obs, env[[2]]$x, env[[2]]$y, env[[2]]$lo, env[[2]]$hi)
colnames(df.env)<- c("obs", "x", "y", "lo", "hi")



# QUEROMOS QUE TODOS OS PONTOS ESTEJAM DENTRO DAS "BANDAS" DE CONFIANCA

df.env %>% ggplot(aes(x=x,y=y)) + 
  geom_point() + 
  geom_line(aes(x=x,y=lo), linetype="dashed") +
  geom_line(aes(x=x,y=hi), linetype="dashed") +
  xlab("Resíduos Simulados") + 
  ylab("Resíduos Observados") + 
  theme_bw()



##################################
### DAQUI EM DIANTE É OPCIONAL ###
##################################

##### VARIAVEIS COM "ZERO VARIANCE" #####

zv_variable<- NULL

for(j in 2:ncol(df)){
  
  if(sd(df[,j])==0){
    zv_variable<- c(zv_variable,names(df)[j])
  }
  
}

zv_variable



##### AGRUPANDO "VARIAVEIS IGUAIS" #####


df_aux<- df %>% dplyr::select(-all_of(c("grave",zv_variable)))
variaveis_modelo_final<- names(modelo_step$coefficients)[-1]

variaveis_iguais<- list()
for(i in 1:length(variaveis_modelo_final)){
  variaveis_iguais[[i]]<- c(variaveis_modelo_final[i],
                            names(df_aux)[abs(cor(df_aux,
                                                  df_aux %>%
                                                    select(variaveis_modelo_final[[i]])))==1])
  variaveis_iguais[[i]]<- unique(variaveis_iguais[[i]])
}

variaveis_iguais




