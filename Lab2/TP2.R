#TP2 d'Analyse de Fisher

#A - Multivariate data set : Fisher Iris

#1)

F = read.csv("C:/Users/victo/OneDrive - ISEP/ISEP/A2/Analyse de données 2018/TP2/iris.data",sep=",")

#2)

var1 = F$SepalLength
hist(var1,20)
var2 = F$SepalWidth
hist(var2)
var3 = F$PetalLength
hist(var3)
var4 = F$PetalWidth
hist(var4)

#3)

var1Mean = mean(var1)
var2Mean = mean(var2)
var3Mean = mean(var3)
var4Mean = mean(var4)

Cov12 = (1/149)*sum((var1-var1Mean)*(var2-var2Mean))

Cov13 = (1/149)*sum((var1-var1Mean)*(var3-var3Mean))

Cov14 = (1/149)*sum((var1-var1Mean)*(var4-var4Mean))

Cov23 = (1/149)*sum((var2-var2Mean)*(var3-var3Mean))

Cov24 = (1/149)*sum((var2-var2Mean)*(var4-var4Mean))

Cov34 = (1/149)*sum((var3-var3Mean)*(var4-var4Mean))

r12 = Cov12/(sd(var1)*sd(var2))

r13 = Cov13/(sd(var1)*sd(var3))

r14 = Cov14/(sd(var1)*sd(var4))

r23 = Cov23/(sd(var2)*sd(var3))

r24 = Cov24/(sd(var2)*sd(var4))

r34 = Cov34/(sd(var3)*sd(var4))

#4)

corR12 = cor(var1,var2)
corR13 = cor(var1,var3)
corR14 = cor(var1,var4)
corR23 = cor(var2,var3)
corR24 = cor(var2,var4)
corR34 = cor(var3,var4)

var12Plot = plot(var1,var2)
var13Plot = plot(var1,var3)
var14Plot = plot(var1,var4)
var23Plot = plot(var2,var3)
var24Plot = plot(var2,var4)
var34Plot = plot(var3,var4)

#5) Intervalle de confiance à 95%

corIris = cor(F[,-5])

sZ = sqrt(1/(150-3))

Z = (log(1+corIris)-log(1-corIris))/2
ZInf = Z-1.96*sZ
ZSup = Z+1.96*sZ
inf = (exp(2*ZInf)-1)/(exp(2*ZInf)+1)
sup = (exp(2*ZSup)-1)/(exp(2*ZSup)+1)



#B - Multivariate data set : Anthropometric data

#1)

M = read.csv("C:/Users/victo/OneDrive - ISEP/ISEP/A2/Analyse de données 2018/TP2/mansize.csv",sep=";")

#2)

summary(M)

#3)

Age = M$Age
hist(Age,breaks = seq(18,24,1))
Heigth = M$Height..cm.
hist(Heigth,breaks = seq(150,204,2))
Weigth = M$Weight..kg.
hist(Weigth,breaks = seq(40,116,2))
Femur = M$Femur.Length..cm.
hist(Femur,breaks = seq(37,63,1))
Arm = M$Arm.span..cm.
hist(Arm,breaks = seq(159,207,2))
Hand = M$Hand.length..cm.
hist(Hand,breaks = seq(15,23,0.5))
Cranial = M$Cranial.volume..cm3.
hist(Cranial,breaks = seq(1298,1558,10))
Penis = M$Penis.size..cm.
hist(Penis,breaks = seq(9,19,0.5))


#4)

corM = cor(M)
plotcorM = plot(M)


#5)

sZ2 = sqrt(1/(161-3))

Z2 = (log(1+corM)-log(1-corM))/2
Z2Inf = Z2-1.96*sZ2
Z2Sup = Z2+1.96*sZ2
inf2 = (exp(2*Z2Inf)-1)/(exp(2*Z2Inf)+1)
sup2 = (exp(2*Z2Sup)-1)/(exp(2*Z2Sup)+1)

#6) 

determinationM = cor(M)*cor(M)


#C - Chi-squared test of independence and categorial variables

#1)

W = read.csv("C:/Users/victo/OneDrive - ISEP/ISEP/A2/Analyse de données 2018/TP2/weather.csv",sep=";")

outlook = table(W$Outlook)
temperature = table(W$Temperature)
humidity = table(W$Humidity)

barplot(outlook, main = "Outlook", xlab = "Outlook", ylab = "Number of cities")
barplot(temperature, main = "Temperature", xlab = "Temperature", ylab = "Number of cities")
barplot(humidity, main = "Humidity", xlab = "Humidity", ylab = "Number of cities")

#2)

contTable1 = table(outlook, temperature)

#r = 4 et c = 3 donc deg = 3*2 = 6

#3)

chisq1 = chisq.test(contTable1)

#P value élevée donc on ne peut pas rejeter l'indépendance des deu variables
#Calcul de Cramer non utile
#Cramer


#4)



contTable2 = table(outlook,humidity)
chisq2 = chisq.test(contTable2)

#P value très faible donc on rejette l'indépendance
#Cramer

Vouthum = sqrt((68.49)/(193*2))
#V = 0.4212306
#On ne peut pas trop conclure



contTable3 = table(temperature,humidity)
chisq3 = chisq.test(contTable3)

#P value très faible donc on rejette l'indépendance
#Chuprov

ptemphum = sqrt((10.331)/(193*sqrt(4)))
#p = 0.1635978
#Elles sont indépendantes

