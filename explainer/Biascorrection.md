## Bridging climate projections and local forecasts

While climate models project the overall development of the climate-related circulations in the earth system, many questions about future developments refer to very specific locations and detailed forecasts about landscape states and weather. This scaling issue poses a challenge to many climate change related projects.

### What is the situation?
It is commonly known that weather forecasts for longer time-spans are likely to be imprecise. It is also no secret that the general weather situation can be predicted much better than a precise timing and amount of rainfall. Moreover, most people have experiences about local deviations of certain weather situations from a forecast (different temperature, rain shadow effects of hills and mountains, etc.).

In climate predictions, climate models (coupled atmosphere ocean general circulation models or GCMs in brief) generally operate with a very similar set of physics like weather models. However, they strongly differ in scale and in the possibility to update their predictions based on observations.

#### Weather models
Weather models operate at a surface resolution of about 9 km (there are models with coarser and finer resolution, too). And they constantly update their predictions to surface-based and satellite-based observations. Hence they can take advatange of data about the development of a pressure system in real time. Still, the next day forecast is contains uncertainty - especially when several pressure systems interact.

#### Climate models
GCMs operate at a surface resolution of about 100 km. This means that a single cell in a GCM averages the conditions and dynamics of the whole German North Sea and its coast (as one example). To get closer to the resolution we actually need to evaluate local impacts, regional climate models (RCMs) are used to re-evaluate the output of GCMs at a much finer resolution of about 12.5 km in our cases. Also with respect to the operable time-step there are differences. Most GCMs and RCMs output monthly and daily data - as they are intended for analyses of the general development of the climate. 

The main difference to weather models however is, that there naturally are no data about the atmosphere to update the forecast to. Hence, climate models are trained with past weather data and then run freely into the future. 

#### Bias correction
This scaling issue and the "free running" leads to deviations from locally observed weather. One straight forward example is local air temperature. The GCM or RCM outputs are likely to be fully plausible but simply a little too high or too low in comparison to a weather station. As these deviations appear as some sort of overall offset between climate forecasts and local observations, and because real-world adaptation decisions require a solid data basis, different techniques to correct or calibrate this apparent bias have been developed.

In general, the operations aim to adjust the observed and modelled data in terms of their overall distributions in a common time span. Bias correction methods analyse how to reproject the model outputs to the observations to minimise this deviation and build a set of rules. These rules are then applied to the full time series of the climate model projections.

Applying such techniques is still under debate since obviously the best way would be to have better models which do not require for bias correction. It is also argued that it is uncertain to what degree climate models can be regarded capable to correctly forecast changes in local weather, if bias correction is necessary. Likewise it is debated if an application of bias correction does skew the model outcomes too much towards the currently observed situation dampening potentially correctly predicted dynamics. 

In the RUINS project, we have to rely on the best data and methods currently available. Moreover, we treat climate forecasts as weather inputs for further models. Hence, we applied a scaled distribution mapping method after Switanek et al. 2017 [(DOI: 10.5194/hess-21-2649-2017)](https://doi.org/10.5194/hess-21-2649-2017) to observations at meteorological stations of the German Weather Service (DWD). 

