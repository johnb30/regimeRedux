# Script to combine different regime datasets into labels
# for supervised topic modeling

# Clear workspace and set path
rm(list=ls())
baseData='~/Dropbox/Research/WardProjects/regimeClassif/Data'

# Helpful functions
library(foreign)
library(countrycode)
library(Hmisc)
char=function(x){as.character(x)}
num=function(x){as.numeric(char(x))}

### Cntries in JSON texts
setwd(paste0(baseData, '/Components'))
cntries=char(read.csv('cntriesForAnalysis.csv', header=F)[,1])
cntries=toupper(cntries)

# Generate consistent country names and subset data
# to relevant period of analysis
prepData=function(data, cvar, yvar, repError, 
	yrS, yrE, detail, keepvars, removeNAs, duplCheck){
	data=data[which(data[,yvar] %in% yrS:yrE),]
	cname=countrycode(data[,cvar],'country.name','country.name')
	cname=toupper(cname)
	data=cbind(data, cname=cname)	
	if(sum(is.na(cname))>0 & repError){
		cat('Unrecognized country names:\n')
		print(data[which(is.na(data$cname)), c(cvar,yvar,'cname')])
		} else {
			subdata=data[which(data$cname %in% cntries),]
			if(detail){
				cat('Following countries dropped:\n')
				print(setdiff(unique(data$cname),unique(subdata$cname)))
			}
			if(removeNAs){
				subdata=na.omit(subdata[,c('cname',cvar,yvar,keepvars)])
			}
			if(duplCheck){
				cat('\nDuplicates:\n')
				print(table(paste0(subdata$cname,subdata[,yvar]))
					[table(paste0(subdata$cname,subdata[,yvar]))>1])
			}
			subdata$id=paste0(subdata$cname,subdata[,yvar])
			subdata[,c('id','cname',cvar,yvar,keepvars)]
	}
 }

# Check for dupes in column of dataframe
dupeCheck=function(x){table(x$id)[table(x$id)>1]}

# Annoying countrycode cases
congoRep=toupper(countrycode('Republic of Congo', 'country.name', 'country.name'))
congoDem=toupper(countrycode('Democratic Republic of Congo', 'country.name', 'country.name'))
nKorea=toupper(countrycode('Korea, North', 'country.name', 'country.name'))
sKorea=toupper(countrycode('Korea, South', 'country.name', 'country.name'))

#############################################
##### Start Democracy Datasets #####
### DD (ends at 2008)
setwd(paste0(baseData,'/regimeData/DD'))
dd=read.csv('dd.csv')
dd$ctryname=char(dd$ctryname)
dd$ctryname[dd$ctryname=='Congo (Brazzaville, Republic of Congo)']=congoRep
dd$ctryname[dd$ctryname=='North Korea']=nKorea
ddFin=prepData(dd, 'ctryname', 'year', TRUE, 
	1999, 2010, TRUE, c('democracy'), TRUE, TRUE)
dupeCheck(ddFin)

### BMR (ends at 2007)
setwd(paste0(baseData,'/regimeData/BMR'))
bmr=read.dta('democracy.dta')
bmr$country=char(bmr$country)
bmr$country[bmr$country=='LIECHSTENSTEIN']='liechtenstein'
bmr$country[bmr$country=='UNITED ARAB E.']='United Arab Emirates'
bmr$country[bmr$country=='EQUATORIAL G']='EQUATORIAL GUINEA'
bmr$country[bmr$country=='GUINEA-BISS']='GUINEA-BISSAU'
bmr$country[bmr$country=='SAMOA, W']='SAMOA'
bmr$country[bmr$country=='PAPUA N.GUINEA']='PAPUA New GUINEA'
bmrFin=prepData(bmr, 'country', 'year', TRUE, 
	1999, 2010, TRUE, c('democracy'), TRUE, TRUE)
dupeCheck(bmrFin)

### FH (ends at 2013)
setwd(paste0(baseData,'/regimeData/Other'))
fh=read.csv('fhdata.csv')
fh$country=char(fh$country)
fh$country[fh$country=='Congo (Kinshasa)']=congoDem
fh$country[fh$country=='Congo (Brazzaville)']=congoRep
fh$country[fh$country=='North Korea']=nKorea
fhFin=prepData(fh, 'country', 'year', TRUE,
	1999, 2013, TRUE, c('Status'), TRUE, TRUE)
fhFin$cname=char(fhFin$cname)
fhFin$cname[fhFin$country=='South Sudan']='SOUTH SUDAN'
fhFin$id=paste0(fhFin$cname, fhFin$year)
dupeCheck(fhFin)

### Polity (ends at 2013)
pol=read.csv('p4v2013.csv')
pol$country=char(pol$country)
pol$country[pol$country=='UAE']='United Arab Emirates'
pol$country[pol$country=='Congo Brazzaville']=congoRep
pol$country[pol$country=='Congo Kinshasa']=congoDem
polFin=prepData(pol, 'country', 'year', TRUE,
	1999, 2013, TRUE, c('polity2'), TRUE, TRUE)
polFin$cname=char(polFin$cname)
polFin$cname[polFin$country=='South Sudan']='SOUTH SUDAN'
polFin$id=paste0(polFin$cname, polFin$year)
polFin$drop=0
if(polFin['16484',1]=='SERBIA2006'){polFin['16484','drop']=1}
if(polFin['13905',1]=='SUDAN2011'){polFin['13905','drop']=1}
polFin=polFin[which(polFin$drop==0),1:(ncol(polFin)-1)]
dupeCheck(polFin)

### Create democracy label
lapply(list(fhFin,polFin),function(x) FUN=dim(x))
demData=merge(fhFin, polFin[,c(1,5)], by='id')
dupeCheck(demData)

demData$democ=0
demData$democ[
	which( demData$Status=='F' & 
		   demData$polity2==10 ) ] = 1

### Polity only dem data with different cuts
demData$polGe10=as.numeric(demData$polity2==10)
demData$polGe9=as.numeric(demData$polity2>=9)
demData$polGe8=as.numeric(demData$polity2>=8)
demData$polGe7=as.numeric(demData$polity2>=7)
demData$polGe6=as.numeric(demData$polity2>=6)
summary(demData)

### Create polity cats
demData$polCat4=NA
demData$polCat4[demData$polity2>=6]=4
demData$polCat4[demData$polity2>=1 & demData$polity2<6]=3
demData$polCat4[demData$polity2<=0 & demData$polity2>-6]=2
demData$polCat4[demData$polity2<=-6]=1
table(demData$polity2, demData$polCat4)

demData$polCat3=cut(demData$polity2, 3, include.lowest=TRUE, labels=1:3)
table(demData$polity2, demData$polCat3)

demData$polCat7=cut(demData$polity2, 7, include.lowest=TRUE, labels=1:7)
table(demData$polity2, demData$polCat7)
##### End of Democracy Datasets #####
#############################################

#############################################
##### Start Mon,Mil,Party Datasets #####
### ARD (ends at 2010)
setwd(paste0(baseData,'/regimeData/ARD'))
ard=read.csv('ard.csv')
ard=ard[which(!is.na(ard$cowcode)),]
ard$country=char(ard$country)
ard$country[ard$country=='Yugoslavia, FR (Serbia/Montenegro)']='Serbia'
ard$country[ard$country=='Congo, Rep.(Brazzaville)']=congoRep
ardFin=prepData(ard, 'country', 'year', TRUE, 
	1999, 2010, TRUE, c('mon','mil','onep'), TRUE, TRUE)
ardFin$drop=0
if(ardFin['6511',1]=='SERBIA2006'){ardFin['6511','drop']=1}
ardFin=ardFin[which(ardFin$drop==0),1:(ncol(ardFin)-1)]
dupeCheck(ardFin)

### GWF (ends at 2010)
setwd(paste0(baseData,'/regimeData/GWF'))
gwf=read.dta('GWF_AllPoliticalRegimes.dta')
gwf$gwf_country=char(gwf$gwf_country)
gwf$gwf_country[gwf$gwf_country=='Luxemburg']='Luxembourg'
gwf$gwf_country[gwf$gwf_country=='UAE']='United Arab Emirates'
gwf$gwf_country[gwf$gwf_country=='Congo/Zaire']=congoDem
gwf$gwf_country[gwf$gwf_country=='Congo-Brz']=congoRep
#gwfFin=prepData(gwf, 'gwf_country', 'year', TRUE, 
	#1999, 2010, TRUE, c('gwf_monarchy','gwf_military','gwf_party'), 
	#TRUE, TRUE)
gwfFin=prepData(gwf, 'gwf_country', 'year', TRUE, 
	1999, 2010, TRUE, c('gwf_monarchy','gwf_military','gwf_party',
                        'gwf_personal'), 
	TRUE, TRUE)
dupeCheck(gwfFin)

### Create mon,mil,party labels
lapply(list(ardFin,gwfFin),function(x) FUN=dim(x))
mmpData=merge(gwfFin, ardFin[,c(1,5:7)], by='id')
dupeCheck(mmpData)

mmpData$monarchy=0
mmpData$monarchy[
	which( mmpData$gwf_monarchy==1 &
		   mmpData$mon==1 ) ] = 1

mmpData$military=0
mmpData$military[
	which( mmpData$gwf_military==1 &
		   mmpData$mil==1 ) ] = 1

mmpData$party=0
mmpData$party[
	which( mmpData$gwf_party==1 &
		   mmpData$onep==1 ) ] = 1

mmpData$personal=0
mmpData$personal[
	which( mmpData$gwf_personal==1) ] = 1
##### End of Mon,Mil,Party Datasets #####
#############################################

#############################################
##### Examine by-year dist of labels #####
library(doBy)
summaryBy(democ ~ year, data=demData, FUN=mean)
summaryBy(monarchy ~ year, data=mmpData, FUN=mean)
summaryBy(military ~ year, data=mmpData, FUN=mean)
summaryBy(party ~ year, data=mmpData, FUN=mean)
##### End examine #####
#############################################

#############################################
##### Save #####
setwd(paste0(baseData,'/regimeData'))
write.csv( demData[,c('cname','year',
	names(demData)[6:ncol(demData)])], 
	'demData_99-13.csv')
write.csv( mmpData[,c('cname','year',
	'monarchy','military','party', 'personal')], 
	'mmpData_99-10.csv')
##### Done #####
#############################################
