# Intro to R Code Examples

# Atomic Classes
## Number Types
n.1 <- 123
class(n.1)
n.2 <- 123L
class(n.2)
n.3 <- 3 + 3i
class(n.3)
class(n.1+n.2)
class(n.2+n.3)

## Character
a <- 'H'
b <- "Hello, World"
class(a)
class(b)

## Logical
a <- T
b <- FALSE

## Formula
f <- Sepal.Length ~ Species
class(f)

# Vectors
a <- c(1, 2, 3)
a
class(a)
b <- c('a', 'b', 'c')
b
class(b)
c <- c(4, 'a', 5)
print(c)
class(c)
length(a)
a[1]
b[2:3]
c[c(1,3)]
c[c(T,F,T)]

## Factor vector example
v <- c("New York", "Chicago", "Seattle", "San Jose", "Gary", "Seattle", 
       "Seattle", "San Jose", "New York", "New York", "New York")
v
class(v)
v.factor <- as.factor(v) # This is an inline comment
v.factor
levels(v.factor) <- c("Chicago", "Gary", "Brooklyn", "San Jose", "Seattle")
print(v.factor)
length(v.factor)
v.factor == "Seattle"

## Matrix
m.c <- cbind(b,c)
m.r <- rbind(b,c)
class(m.c)
dim(m.c)
colnames(m.c)
rownames(m.c)
m.c
m.c[,'b']
m.c[1,2]
m.r[1:2,2:3]
m.c[m.c[,1]=='a',]
m.c[m.c[,2]=='a',]
m.c[m.c=='a']

## Data frame
data(iris)
class(iris)
names(iris)
nrow(iris)
ncol(iris)
dim(iris)
head(iris)
tail(iris)
str(iris)
summary(iris)
head(iris[,"Sepal.Length"])
head(iris[,c("Petal.Length", "Petal.Width")])
head(iris[,1])
head(iris[,1:3])
head(iris[,-c(1,2)])
head(iris[iris$Species=="virginica",])

## List
l.1 <- list(1, 'a', TRUE, 1+4i)
l.2 <- list(a=c(T,T,F,F), b=2, hello="World")
l.1
l.2
l.2$a
l.2$hello
l.2[[1]]
l.2[1]
l.2$func <- summary
l.2$func(l.2)

# Basic Language Features
## Install and load packages
install.packages('plyr')
library(plyr)

## Directory navigation
getwd()
setwd('Introduction_to_R_Programming')
getwd()
dir()
source('source_example.R')
setwd('..')
getwd()

## Reading and writing data
iris.data <- read.csv('Datasets/Iris_Data.csv')
iris.data.2 <- read.table('Datasets/Iris_Data.csv')
head(iris.data)
iris.out <- iris.data
iris.out$Sepal.Length <- log(iris.out$Sepal.Length)
head(iris.out)
write.csv(iris.out, 'iris_out.csv')


## Flow Control Structures
a <- 23
if( a == 23 ){
    print("First if")
} else if (a == 25) {
    print("else if")
} else {
    print("else")
}
a <- 25
if( a == 23 ){
    print("First if")
} else if (a == 25) {
    print("else if")
} else {
    print("else")
}
a <- 2
if( a == 23 ){
    print("First if")
} else if (a == 25) {
    print("else if")
} else {
    print("else")
}

### Don't use these loops
### for( lev in v.factor ) { }
### while (a == 2) {a <- 3}

## Functions