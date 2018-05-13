# Rede Neural de Múltiplas Camadas para Regressão Simbólica

## Projeto de Graduação em Computação UFABC

21008214 Maira Zabuscha de Lima

### Testes
Eu testei pras mesmas bases de dados do programa em c e houveram duas q bugam (bioavailability e ppb)

`File "/home/ufabc/anaconda3/lib/python3.6/site-packages/scipy/interpolate/interpolate.py", line 528, in __init__
    "least %d entries" % minval)
ValueError: x and y arrays must have at least 2 entries`

O restante deu rmse bom, todas quiseram o modelo lassoLarsCV.

Seus resultados eram:

`| dataset         | MAE_MLP        | RMSE_MLP       | MAE_XG         | RMSE_XG       |
|-----------------|----------------|----------------|----------------|---------------|
| airfoil         | 6.18002242479  | 7.67184518627  | 1.11938559178  |1.8309682719   |
| bioavailability | 20.5488266959  | 25.3781544225  | 24.4211684128  |31.7162529705  |
| concrete        | 8.29282257088  | 10.4776342927  | 2.87123275074  |4.0740942837   |
| cpu             | 80.5331905948  | 222.712701917  | 22.7338762755  |77.5892865368  |
| energyCooling   | 2.52266654687  | 3.53526528505  | 0.361799431505 |0.514431391528 |
| energyHeating   | 2.54868218149  | 3.42836981878  | 0.222104601588 |0.338121978888 |
| forestfires     | 22.7982665304  | 107.530502684  | 29.0364162179  |116.625945888  |
| ppb             | 28.6142294122  | 33.5858995656  | 31.8702452249  |40.1568089435  |
| towerData       | 18.8099924172  | 25.4063703414  | 11.8206282622  |17.0157431156  |
| wineRed         | 0.479160333719 | 0.626380925203 | 0.378293441273 |0.591409131066 |
| wineWhite       | 0.59194468634  | 0.76440593106  | 0.452436536478 |0.678575144104 |
| yacht           | 7.31679187865  | 9.67918507725  | 0.392170667968 |0.849322392096 |`

### Resultados

`n inter 5

lassoCV 6.40009411517889
lassoLarsCV 5.2220226976922905
min lassoLarsCV
rmse airfoil 5.357374009772064 

lassoCV 15.477505972484359
lassoLarsCV 14.45235554319056
min lassoLarsCV
rmse concrete 14.66909108429513 

lassoCV 559.8124571040972
lassoLarsCV 290.97999979923344
min lassoLarsCV
rmse cpu 70.06103712444842 

lassoCV 9.739216398637595
lassoLarsCV 5.137992183801275
min lassoLarsCV
rmse energyCooling 5.5709338418284835 

lassoCV 8.666101347456971
lassoLarsCV 4.081405742994017
min lassoLarsCV
rmse energyHeating 4.5585815006219 

lassoCV 21.61956429670779
lassoLarsCV 19.3531125546516
min lassoLarsCV
rmse forestfires 82.25011717728836 

lassoCV 75.86265996768928
lassoLarsCV 61.75929799072116
min lassoLarsCV
rmse towerData 66.07240206253448 

lassoCV 0.7460803807618895
lassoLarsCV 0.6975787410971579
min lassoLarsCV
rmse wineRed 0.7330427458252987 

lassoCV 0.8822215317407687
lassoLarsCV 0.8595346002337939
min lassoLarsCV
rmse wineWhite 0.8663455024818917 

lassoCV 10.568085045501523
lassoLarsCV 10.16122016227372
min lassoLarsCV
rmse yacht 9.731068516319386`