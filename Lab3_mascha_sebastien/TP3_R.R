#TP3 d'Analyse de données

#A - Step by step linear regression using the ”bats” data

#A.1 Correlation circles

#1)

tab = read.table("C:/Users/victo/OneDrive - ISEP/ISEP/A2/Analyse de données 2018/TP3/tabBats.txt",sep=" ", header = TRUE)


#2)

tab


#3)

str(tab)
tab[,1:3]=NULL


#4)

library(FactoMineR)
result <- PCA(tab)


#5) Voir compte rendu



#A.2 First linear regression using R

#1)

plot(tab$BOW,tab$BRW)

cov_BOW_BRW = cov(tab$BOW,tab$BRW)
var_BOW = var(tab$BOW)
sd_BOW = sd(tab$BOW)
sd_BRW = sd(tab$BRW)

b_1_BOW_BRW = cov_BOW_BRW/var_BOW
b_1_BOW_BRW = 8.999878


mean_BOW = mean(tab$BOW)
mean_BRW = mean(tab$BRW)

b_0_BOW_BRW = mean_BRW-(b_1_BOW_BRW*mean_BOW)


#2)

mod=lm(tab$BRW~tab$BOW)

#3)

plot(mod)


#4)

summary(mod)

#5)

plot(tab$BOW,tab$BRW)
abline(coef(mod),col="red")



#A.3 Second linear regression

#1)

tab2 = tab[-14,]

plot(tab$BOW,tab$BRW)
plot(tab2$BOW,tab2$BRW)

library(FactoMineR)
result <- PCA(tab2)

#2)

mod2=lm(tab2$BRW~tab2$BOW)
plot(mod2)
summary(mod2)
plot(tab2$BOW,tab2$BRW)
abline(coef(mod2),col="red")

#3)

plot(tab$BOW,tab$BRW)
abline(coef(mod),col="red")
abline(coef(mod2),col="blue")



#B - Application to mansize dataset


#1)

M = read.csv("C:/Users/victo/OneDrive - ISEP/ISEP/A2/Analyse de données 2018/TP2/mansize.csv",sep=";")

library(FactoMineR)
result <- PCA(M)


#2)

mod3=lm(M$Height..cm.~M$Femur.Length..cm.)
plot(mod3)
summary(mod3)
plot(M$Height..cm.~M$Femur.Length..cm.)
abline(coef(mod3),col="red")

#3) Voir rapport