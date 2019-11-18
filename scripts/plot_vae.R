#plot ag1000g locator vae tests
setwd("~/Dropbox/popvae/")
library(ggplot2);library(data.table);library(plyr);library(sp);library(magrittr);library(cowplot)
theme_set(theme_classic()+
            theme(axis.text = element_text(size=6),
                  axis.title=element_text(size=7),
                  legend.text = element_text(size=6),
                  legend.title=element_text(size=7),
                  strip.background = element_blank(),
                  strip.text = element_text(size=6)))

a <- fread("vaetest.txt",data.table=F,header=T)
b <- fread("vaetest_pca.txt",data.table=F)
b$model <- "PCA"
a$model <- "VAE"
names(a) <- names(b)
a <- rbind(a,b)
locs <- fread("data/hgdp/hgdp_sample_data.txt")
a <- merge(a,locs,by="sampleID")


#latent space
dimplot <- ggplot(data=a,aes(x=PC1,y=PC2,col=region))+
  #scale_color_brewer(palette="Set1")+
  theme(legend.key.height = unit(4,"mm"))+
  facet_wrap(~model,nrow=1,scales="free")+
  geom_point(shape=1,size=.5,stroke=0.4)+#stat_ellipse()+
  xlab("Axis 1")+ylab("Axis 2")+
  guides(color=guide_legend(override.aes = list(size=3,stroke=1.5)))


#distance comparisons
dists <- data.frame(vae_dist=c(spDists(as.matrix(a[a$model=="VAE",c("PC1","PC2")]))),
                    pca_dist=c(spDists(as.matrix(a[a$model=="PCA",c("PC1","PC2")]))),
                    geo_dist=c(spDists(as.matrix(a[a$model=="PCA",c("longitude","latitude")]))))
vae_dist <- spDists(as.matrix(a[a$model=="VAE",c("PC1","PC2")]))

combos <- combn(unique(a$sampleID),2)[,1:1000] #get all combinations of indivkiduals 
dists <- data.frame(ind1=character(),
                    ind2=character(),
                    vae_dist=numeric(),
                    PC_dist=numeric(),
                    geo_dist=numeric(),
                    stringsAsFactors = F)[0,] #create df for output 
for(i in 1:ncol(combos)){ #loop through pairs and bind to output df
  vae_dist <- dist(a[(a$model=="VAE" & a$sampleID %in% combos[,i]),c("PC1","PC2")])[1]
  PC_dist <- dist(a[(a$model=="PCA" & a$sampleID %in% combos[,i]),c("PC1","PC2")])[1]
  latlongs <- as.matrix(a[(a$model=="PCA" & a$sampleID %in% combos[,i]),c("longitude","latitude")])
  geo_dist <- spDistsN1(t(as.matrix(latlongs[1,])),t(as.matrix(latlongs[2,])),longlat=T)[1]
  ind1 <- combos[1,i]
  ind2 <- combos[2,i]
  row <- data.frame(ind1,ind2,vae_dist,PC_dist,geo_dist,stringsAsFactors = F)
  dists <- rbind(dists,row)
}

#add columns showing regions for each pair
tmp <- merge(dists,locs[,c("sampleID","region","population")],by.x="ind1",by.y="sampleID")
tmp <-merge(tmp,locs[,c("sampleID","region","population")],by.x="ind2",by.y="sampleID")
names(tmp)[6:9] <- c("region_ind1","pop_ind1","region_ind2","pop_ind2")
tmp$pop_comparison <- paste(tmp$pop_ind1,tmp$pop_ind2,sep="x")



r2vae <- lm(vae_dist~geo_dist,data=dists) %>% summary() %>% .$adj.r.squared
r2pca <- lm(pca_dist~geo_dist,data=dists) %>% summary() %>% .$adj.r.squared
dists <- melt(dists,id.vars=c("geo_dist"))
dists$r2 <- NA
dists$r2[dists$variable=="vae_dist"] <- r2vae
dists$r2[dists$variable=="pca_dist"] <- r2pca
dists$variable <- factor(dists$variable,levels=c("pca_dist","vae_dist"))
print(r2vae)
print(r2pca)

distplot <- ggplot(data=dists[sample(1:nrow(dists),1e4),],aes(x=geo_dist,y=value))+
  facet_wrap(~variable,scales="free")+
  ylab("latent space distance")+xlab("geographic distance")+
  geom_point(shape=1,alpha=0.5,size=0.5)+
  geom_smooth(method="lm",se=F,color="red")

#VAE latent space over the training routine
files <- list.files("training_preds/",full.names = T) %>% grep(".txt",.,value=T)
pd <- fread(files[1],header=T);pd$epoch <- 0;pd <- pd[0,]
for(f in files){
  tmp <- fread(f,header=T)
  tmp$epoch <- as.integer(gsub("[[:alpha:]]|\\.","",basename(f)))
  pd <- rbind(pd,tmp)
}
names(pd) <- c("x","y","sampleID","epoch")
pd <- merge(pd,locs,by="sampleID")
ggplot(data=pd,aes(x=x,y=y,col=region))+
  facet_wrap(~epoch,scales="free")+
  geom_point(shape=1,size=0.7)+
  guides(color=guide_legend(override.aes = list(size=3,stroke=2)))

history <- fread("vae_history.txt",data.table=F)
history$epoch <- 1:nrow(history)
ggplot(data=history,aes(x=epoch,y=loss))+
  geom_line()

pdf("fig/vaetest_hgdp.pdf",width=5,height=4,useDingbats=F)
ggdraw()+
  draw_plot(dimplot,0,0.45,1,0.55)+
  draw_plot(distplot,0,0,0.9,0.45)
dev.off()

