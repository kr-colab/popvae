library(ggplot2);library(reshape);library(plyr);library(data.table);library(raster);library(broom);library(cowplot)
setwd("~/Dropbox/popvae/")
theme_set(theme_classic()+theme(text=element_text(size=8),
                                strip.background = element_blank()))

######## ag1000g ##########
a <- fread("out/ag1000g_phase1_800kSNPs_latent_coords.txt",header=T)
names(a) <- c("LD1","LD2","sampleID")
pc <- fread("out/ag1000g_phase1_100kSNPs_pca.txt")[,c(1,2,21)]
names(pc) <- c("LD1","LD2","sampleID")
b <- fread("~/locator/data/ag1000g/samples.all.txt")
b$sampleID <- b$ox_code
c <- merge(a,b,by="sampleID")
d <- merge(b,pc,by="sampleID")
c$method <- "VAE"
d$method <- "PCA"
e <- rbind(c,d)

#final prediction
ggplot(e,aes(x=LD1,y=LD2,fill=country))+
  facet_wrap(~method,scales="free")+
  geom_point(stroke=0.2,shape=21)+
  scale_fill_brewer(palette = "Paired",name="Country")


########## HGDP ###########
a <- fread("out/hgdp_chr1_1e5snps_latent_coords.txt",header=T)
names(a) <- c("LD1","LD2","sampleID")
pc <- fread("out/hgdp_chr1_pca.txt")[,c(1,2,21)]
names(pc) <- c("LD1","LD2","sampleID")
b <- fread("data/hgdp/hgdp_sample_data.txt")
c <- merge(a,b,by="sampleID")
d <- merge(b,pc,by="sampleID")
c$method <- "VAE"
d$method <- "PCA"
e <- rbind(c,d)

#final prediction
ggplot(e,aes(x=-LD1,y=-LD2,fill=region))+
  facet_wrap(~method,scales="free")+
  geom_point(shape=21,stroke=0.2)+
  scale_fill_brewer(palette = "Paired")

#summarized one latent dimension a map
load("~/locator/locator_hgdp/cntrymap.Rdata")
map <- crop(map,c(-170,170,-50,71))
d <- ddply(c,.(longitude,latitude),summarize,n=length(LD1),mean_LD1=mean(LD1),mean_LD2=mean(LD2))
ggplot()+coord_map(projection="mollweide")+
  theme(axis.title.x=element_blank(),axis.title.y=element_blank(),
        legend.position=c(0.1,0.5),legend.background = element_blank(),
        legend.spacing = unit(0,"mm"))+
  scale_fill_distiller(palette = "YlGnBu",name="Mean LD2")+
  scale_size_continuous(name="Samples")+
  geom_polygon(data=map,aes(x=long,y=lat,group=group),lwd=0.2,col="white",fill="grey80")+
  geom_point(data=d,aes(x=longitude,y=latitude,fill=mean_LD2,size=n),shape=21,stroke=0.2)+ #can flip LD1 v LD2 here
  guides(fill=guide_colorbar(barheight = unit(20,"mm"),barwidth = unit(4,"mm")),
         size=guide_legend(keyheight = unit(2,"mm")))

#predictions during training
library(gganimate)
a2 <- fread("out/hgdp_chr1_1e5snps_training_preds.txt")
names(a2) <- c("LD1","LD2","sampleID","epoch")
b2 <- fread("data/hgdp/hgdp_sample_data.txt")
c2 <- merge(a2,b2,by="sampleID")
c2 <- ddply(c2,.(epoch),function(e) {
  e$LD1 <- e$LD1-mean(e$LD1)
  e$LD2 <- e$LD2-mean(e$LD2)
  return(e)
})
ggplot(c2,aes(x=LD1,y=-LD2,fill=region))+
  #facet_wrap(~epoch,scales="free")+
  geom_point(shape=21,stroke=0.2,size=3)+
  scale_fill_brewer(palette = "Paired")+
  transition_manual(epoch)+
  labs(title='Epoch: {frame}')
anim_save("out/hgdp_val_train_preds.gif")

######### pabu w map #######
a <- fread("out/pabu_latent_coords.txt")
names(a) <- c("x","y","sampleID")
b <- fread("data/pabu/pabu_full_data.csv")
c <- merge(a,b,by="sampleID")
#c$species[is.na(c$species)] <- "unknown"

c$k3clust <- kmeans(c[,c("x","y")],centers=3)[[1]]

ggplot(c,aes(x=x,y=y,fill=factor(k3clust)))+
  geom_point(shape=21,stroke=0.2)+
  scale_fill_brewer(palette = "Dark2")
scale_fill_distiller(palette="YlGnBu")

load("~/locator/locator_hgdp/cntrymap.Rdata")
map <- crop(map,c(-120,-60,10,43))
ggplot()+coord_map()+
  xlim(-120,-60)+ylim(10,43)+
  geom_polygon(data=map,aes(x=long,y=lat,group=group),lwd=0.2,col="white",fill="grey80")+
  geom_point(data=c,aes(x=Longitude,y=Latitude,fill=factor(k3clust)),shape=21,stroke=0.2,size=4)+
  scale_fill_brewer(palette="Dark2")