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

df<- df %>% clean_names()               # nome das variaveis em minusculo
names(df)<- gsub("_1", "", names(df))   # modificando nomes

df<- df %>% filter(!is.na(grave))       # removendo linhas com dados faltantes na resposta
df<- df %>% dplyr::select(-id)          # removendo variavel identificadora
df$grave<- df$grave %>% factor()        # transformando a resposta em fator



##### SPLIT #####

set.seed(0)
split<- initial_split(df, strata=grave, prop=0.8)

# df_train<- training(split)
# df_test<- testing(split)

folds<- vfold_cv(df, v=2, repeats=5, strata=grave)
#folds<- vfold_cv(df_train, v=2, repeats=5, strata=grave)



##### PRÉ-PROCESSAMENTO #####

receita<- recipe(grave ~ . , data = df) %>%                   # modelo e dados
  #step_zv(all_predictors()) %>%                              # variaveis sem variabilidade
  step_nzv(all_predictors(),freq_cut=tune()) %>%              # variaveis quase sem variabilidade
  step_filter_missing(all_predictors(),threshold = 0.4) %>%   # variaveis +40% de faltantes
  #step_normalize(all_numeric_predictors()) %>%               # normalizar variaveis
  #step_impute_knn(all_predictors()) %>%                      # imputando faltantes
  step_naomit() %>%                                           # deletando faltantes
  step_corr(all_numeric_predictors(),threshold=tune())        # removendo variaveis correlacionadas





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
# 332.47 sec elapsed



## VISUALIZANDO OS MELHORES MODELOS (BEST RMSE)

show_best(tune_pls,n=3)




## MODELOS TREINADOS APOS TUNAR OS HIPERPARAMETROS

wf_pls_trained<- wf_pls %>% finalize_workflow(select_best(tune_pls)) %>% fit(df)




## COEFICIENTES DOS MODELOS TREINADOS

k<- 30        # numero de variaveis selecionadas

variaveis_selecionadas<- vip::vi(wf_pls_trained) %>% dplyr::select(Variable)
variaveis_selecionadas<- variaveis_selecionadas$Variable[1:k]

variaveis_selecionadas



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

modelo_step<- stepAIC(modelo_base, direction="both", trace=0)
modelo_step %>% summary()
modelo_step %>% tbl_regression(exponentiate = TRUE,
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




