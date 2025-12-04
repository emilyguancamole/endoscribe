## Methods

### Data Integration

Data from 12 studies were combined to train and build a prediction model. The total number of patients was 7,389. The incidence of PEP was the primary outcome of interest. A standard set of 22 risk factors and 3 treatments (aggressive hydration, indomethacin, and pancreatic duct stent) were considered. 

* include variable definitions


### Statistical model

The statistical analysis was performed using R 4.0.0.

#### Feature selection and imputation

Among the risk factors, pancreatic duct brush cytology and pancreas divisum were removed due to their rare occurrence (< 25 patients out of 7,389 total patients) and type of sod was removed due to high missingness.  Missing values in the remaining risk factors were imputed using median imputation and k-nearest neighbors. Samples in studies 5 (?) and 12 (?) that were missing values in binary variables were first imputed using the median. The rest of the missing values in other samples were then imputed from the 10 nearest neighbors from studies 5 and 12 using the `caret` package.

#### Prediction model

A gradient boosted machine model was fitted using the `caret` package. The number of trees and the interaction depth of the tree was tuned under 5-fold cross-validation over a grid of 50, 100, 150, 500, and 1000 trees and a depth of 1, 2, 3, and 4, while the learning rate was held constant at 0.1 and the minimum number of training set samples in a node to commence splitting was held constant at 10. The model trained over the entire dataset had tuning parameters of 150 trees and an interaction depth of 3. To evaluate the performance of the model, 5-fold cross-validation was performed. Results were compared to a ridge logistic regression model and a random forest classifier.

#### Variable Importance

To assess the marginal importance of each risk factor, we calculated the incidence of PEP for each variable. For example, for continuous variables like age, we calculated the incidence of PEP among those with age greater than the mean age in the dataset. For binary variables like trainee involvement, we calculated the incidence of PEP among cases with trainee involvement. In addition, we rank the variables by the mean decrease in entropy or logistic loss when a tree is split on that variable (Friedman 2001). This represents the contribution of each risk factor towards the predictive value of the model and is an overall metric for the variable importance.

To assess the contribution of each risk factor towards the prediction for each individual, we use local interpretable model-agnostic explanations (LIME) implemented with the `lime` package. LIME trains local surrogate models in the neighborhood around each individual by fitting a local linear ridge regression model to permuted samples and predictions from our gradient boosted tree, weighted by their Euclidean distance to the individual sample. The coefficients of the risk factors in the local linear model are then presented as LIME weights, which approximate the relative importance of each risk factor in determining the predicted risk of PEP for a given individual.

#### Treatment effects

We considered the effects of 4 different treatments: aggressive hydration, indomethacin, indomethacin and aggressive hydration, and pancreatic duct stent. Each treatment effect was estimated using the subset of patients for whom the given treatment has been randomized, with the exception of pancreatic duct stent. 

In particular, to estimate the treatment effect for treatment $t$ and a new patient $i$, the prediction score $p_i$ is first estimated using the full model, with values for all treatment options set to 0. A model $M_t$ is trained on the subset of patients from studies that randomized treatment $t$. Using this model, two new prediction scores are obtained for patient $i$: (1) $p_{it_0}$ which is the prediction score under $M_t$ when patient $i$ did not receive treatment $t$ and (2) $p_{it_1}$, which is the prediction score under $M_t$ when patient $i$ received treatment $t$. The ratio of these two values $a_{it} = \frac{p_{it_1}}{p_{it_0}}$ is the adjustment factor. To account for the case when $p_{it0}$ is very small, causing the adjustment factor to be very large, a shrinkage factor is also applied to $a_{it}$ when $p_{it0} < 0.1$. Specifically, a shrunken adjustment factor is calculated as $a_{it}' = a_{it}*10*\min(p_{it0}, 0.1) + (1-10*\min(p_{it0}, 0.1))$. The prediction score for patient $i$ with treatment $t$ is estimated to be $p_{it} = p_i*a_{it}'$ and the treatment effect is therefore the difference between the two prediction scores $p_{it} - p_i$.

For the indomethacin and aggressive hydration combination treatment, this procedure was performed twice. Since no studies randomized patients to the combination treatment, two adjustment factors were computed, one for indomethacin $a_{it_{indo}}'$ and another for aggressive hydration $a_{it_{ah}}'$. The overall adjustment factor for the indomethacin and aggressive hydration combination is taken as the product $a_{it_{indo}}'*a_{it_{ah}}'$.

* insert table for which studies were used for which treatment



## References

J.H. Friedman (2001). "Greedy Function Approximation: A Gradient Boosting Machine," Annals of Statistics 29(5):1189-1232.