library(ggplot2);library(reshape);library(plyr);
library(data.table);library(raster);library(broom);library(cowplot)

setwd("~/popvae/")
theme_set(theme_classic()+theme(text=element_text(size=8),
                                strip.background = element_blank()))

###########################################################
################# Selasphorus 3-population ################
###########################################################
a <- fread("out/selasphorus_latent_coords.txt",header=T)
names(a) <- c("LD1","LD2","sampleID")
pc <- fread("out/selasphorus_pca.txt")[,c(1,2,21)]
names(pc) <- c("LD1","LD2","sampleID")
b <- fread("data/selasphorus/ddrad_sample_data.txt")
c <- merge(a,b,by="sampleID")
d <- merge(b,pc,by="sampleID")
c$method <- "VAE"
d$method <- "PCA"
e <- rbind(c,d)

pdf("out/selasphorus_3sp_vae_v_pdf.pdf",width=6,height=2.5,useDingbats = F)
ggplot(e,aes(x=LD1,y=LD2,fill=Species))+
  facet_wrap(~method,scales="free")+
  geom_point(stroke=0.2,shape=21,alpha=0.8)+
  scale_fill_brewer(palette = "Set1",name="Species")+
  guides(fill=guide_legend(override.aes = list(size=4)))
dev.off()

load("~/locator/locator_hgdp/cntrymap.Rdata")
map <- crop(map,c(-145,-90,10,62))
d <- ddply(c,.(Longitude,Latitude,Species),summarize,n=length(LD1),mean_LD1=mean(LD1),mean_LD2=mean(LD2))
d <- subset(d,Species!="Selasphorus calliope")
migrants <- subset(c,Species=="Selasphorus rufus" & State %in% c("CA","NM","AZ","UT","TX"))
pdf("out/selasphorus_3sp_map.pdf",useDingbats = F,width=4.5,height=3.5)
ggplot()+coord_map(projection = "mollweide")+
  theme(axis.title.x=element_blank(),axis.title.y=element_blank(),
       # legend.position=c(0.1,0.5),
        legend.background = element_blank(),
        legend.spacing = unit(0,"mm"))+
  scale_fill_distiller(palette = "YlGnBu",name="Mean LD1")+
  scale_size_continuous(name="Samples",breaks=c(1,3,5,7))+
  scale_shape_manual(values=c(21,22,23))+
  geom_polygon(data=map,aes(x=long,y=lat,group=group),lwd=0.2,col="white",fill="grey80")+
  geom_point(data=d,aes(x=Longitude,y=Latitude,fill=mean_LD1,size=n,shape=Species))+ #can flip LD1 v LD2 here
  guides(fill=guide_colorbar(barheight = unit(20,"mm"),barwidth = unit(4,"mm")),
         size=guide_legend(keyheight = unit(2,"mm")),
         shape=guide_legend(override.aes = list(size=4)))
dev.off()

##################################################################
################# Selasphorus full species sample ################
##################################################################

a <- fread("out/selasphorus_full_latent_coords.txt",header=T)
names(a) <- c("LD1","LD2","sampleID")
pc <- fread("out/selasphorus_full_pca.txt")[,c(1,2,21)]
names(pc) <- c("LD1","LD2","sampleID")
b <- fread("data/selasphorus/full_specimen_data.txt")
c <- merge(a,b,by="sampleID")
d <- merge(b,pc,by="sampleID")
c$method <- "VAE"
d$method <- "PCA"
e <- rbind(c,d)

pdf("out/selasphorus_full_vae_v_pdf.pdf",width=6,height=3,useDingbats = F)
ggplot(e,aes(x=LD1,y=LD2,fill=Species))+
  facet_wrap(~method,scales="free")+
  geom_point(stroke=0.4,shape=21,alpha=0.8)+
  scale_fill_brewer(palette = "Paired",name="Species")+
  guides(fill=guide_legend(override.aes = list(size=4),keyheight=unit(1,"mm")))
dev.off()


#######################################
################# PF7K ################
#######################################

a <- fread("out/pf7k_chr14_latent_coords.txt",header=T)
names(a) <- c("LD1","LD2","sampleID")
#a <- subset(a,a$LD2<quantile(a$LD2,0.999)) #outliers seem to be a problem

pc <- fread("out/pf7k_chr14_pca.txt")[,c(1,2,21)]
names(pc) <- c("LD1","LD2","sampleID")
b <- fread("~/locator/data/pf7k/pf7k_sample_data_full.txt")
c <- merge(a,b,by="sampleID")
d <- merge(b,pc,by="sampleID")
c$method <- "VAE"
d$method <- "PCA"
e <- rbind(c,d)
e <- subset(e,population != "Lab")
e$population <- factor(e$population,levels=rev(levels(factor(e$population))))

#final prediction
pdf("out/pf7k_chr14_pca_v_vae.pdf",width=6,height=2.5,useDingbats = F)
ggplot(e,aes(x=LD1,y=LD2,fill=population))+
  facet_wrap(~method,scales="free")+
  geom_point(stroke=0.2,shape=21,alpha=0.7)+
  scale_fill_brewer(palette = "Paired",name="Population")+
  guides(fill=guide_legend(override.aes = list(size=4)))
dev.off()

load("~/locator/locator_hgdp/cntrymap.Rdata")
map <- crop(map,c(-170,170,-50,71))
d <- ddply(c,.(x,y,population),summarize,n=length(LD1),mean_LD1=mean(LD1),mean_LD2=mean(LD2))
ggplot()+coord_map(projection="mollweide")+
  theme(axis.title.x=element_blank(),axis.title.y=element_blank(),
        legend.position=c(0.1,0.5),legend.background = element_blank(),
        legend.spacing = unit(0,"mm"))+
  scale_fill_distiller(palette = "YlGnBu",name="Mean LD1")+
  scale_size_continuous(name="Samples")+
  geom_polygon(data=map,aes(x=long,y=lat,group=group),lwd=0.2,col="white",fill="grey80")+
  geom_point(data=d,aes(x=x,y=y,fill=mean_LD1,size=n),shape=21,stroke=0.2)+ #can flip LD1 v LD2 here
  guides(fill=guide_colorbar(barheight = unit(20,"mm"),barwidth = unit(4,"mm")),
         size=guide_legend(keyheight = unit(2,"mm")))

library(gganimate)
a2 <- fread("out/pf7k_chr14_training_preds.txt")
names(a2) <- c("LD1","LD2","sampleID","epoch")
b2 <- b
c2 <- merge(a2,b2,by="sampleID")
c2 <- ddply(c2,.(epoch),function(e) { #center predictions for prettier plots
  e$LD1 <- e$LD1-mean(e$LD1)
  e$LD2 <- e$LD2-mean(e$LD2)
  return(e)
})
ggplot(c2,aes(x=-LD1,y=-LD2,fill=population))+
  #facet_wrap(~epoch,scales="free")+
  geom_point(shape=21,stroke=0.2,size=3)+
  scale_fill_brewer(palette = "Paired")+
  transition_manual(epoch)+
  labs(title='Epoch: {frame}')
anim_save("out/pf7k_chr14_training_preds.gif")

##########################################
################# ag1000g ################
##########################################
a <- fread("out/ag1000g_phase1_5e5snps_latent_coords.txt",header=T)
names(a) <- c("LD1","LD2","sampleID")
pc <- fread("out/ag1000g_phase1_5e5snps_pca.txt")[,c(1,2,21)]
names(pc) <- c("LD1","LD2","sampleID")
b <- fread("~/locator/data/ag1000g/samples.all.txt")
b$sampleID <- b$ox_code
c <- merge(a,b,by="sampleID")
d <- merge(b,pc,by="sampleID")
c$method <- "VAE"
d$method <- "PCA"
e <- rbind(c,d)

#final prediction
pdf("out/ag1000g_phase1_5e5snps_vae.pdf",width=6,height=2.5,useDingbats = F)
ggplot(e,aes(x=LD1,y=LD2,fill=country))+
  facet_wrap(~method,scales="free")+
  geom_point(stroke=0.2,shape=21)+
  scale_fill_brewer(palette = "Paired",name="Country")+
  guides(fill=guide_legend(override.aes = list(size=4)))
dev.off()


#animated gif of (centered) predictions during training
library(gganimate)
a2 <- fread("out/ag1000g_phase1_5e5snps_training_preds.txt")
names(a2) <- c("LD1","LD2","sampleID","epoch")
b2 <- fread("data/ag1000g/anopheles_samples_sp.txt")
c2 <- merge(a2,b2,by="sampleID")
c2 <- ddply(c2,.(epoch),function(k) { #center predictions for prettier plots
  k$LD1 <- k$LD1-mean(k$LD1)
  k$LD2 <- k$LD2-mean(k$LD2)
  return(k)
})
ranges <- ddply(c2,.(epoch),summarize,r2=max(LD2)-min(LD2),r1=max(LD1)-min(LD1))
dropranges <- subset(ranges,r2==max(ranges$r2)|r1==max(ranges$r1))
c2 <- subset(c2,!(epoch %in% dropranges$epoch))
ggplot(c2,aes(x=LD1,y=LD2,fill=country))+
  geom_point(shape=21,stroke=0.2,size=3)+
  scale_fill_brewer(palette = "Paired")+
  transition_manual(epoch)+
  labs(title='Epoch: {frame}')
anim_save("out/ag1000g_5e5snps.gif")

########################################
################# HGDP #################
########################################

a <- fread("out/hgdp_chr1_1e6snpsv3_latent_coords.txt",header=T)
names(a) <- c("LD1","LD2","sampleID")
pc <- fread("out/hgdp_chr1_1e6snpsv3_pca.txt")[,c(1,2,21)]
names(pc) <- c("LD1","LD2","sampleID")
b <- fread("data/hgdp/hgdp_sample_data.txt")
c <- merge(a,b,by="sampleID")
d <- merge(b,pc,by="sampleID")
c$method <- "VAE"
d$method <- "PCA"
e <- rbind(c,d)

#final prediction
pdf("out/hgdp_chr1_1e6snpsv3_vae_v_pca.pdf",useDingbats = F,width=6,height=2.5)
ggplot(e,aes(x=-LD2,y=-LD1,fill=region))+
  facet_wrap(~method,scales="free")+
  geom_point(shape=21,stroke=0.2)+
  scale_fill_brewer(palette = "Paired")+
  guides(fill=guide_legend(override.aes = list(size=4)))
dev.off()

#summarized one latent dimension a map
pdf("out/hgdp_chr1_1e6snpsv3_vae_map.pdf",useDingbats = F,width=5.5,height=2.75)
load("~/locator/locator_hgdp/cntrymap.Rdata")
map <- crop(map,c(-170,170,-50,71))
d <- ddply(c,.(longitude,latitude),summarize,n=length(LD1),mean_LD1=mean(LD1),mean_LD2=mean(LD2))
ggplot()+coord_map(projection="mollweide")+
  theme(axis.title.x=element_blank(),axis.title.y=element_blank(),
        legend.position=c(0.1,0.5),legend.background = element_blank(),
        legend.spacing = unit(0,"mm"))+
  scale_fill_distiller(palette = "YlGnBu",name="Mean LD1")+
  scale_size_continuous(name="Samples")+
  geom_polygon(data=map,aes(x=long,y=lat,group=group),lwd=0.2,col="white",fill="grey80")+
  geom_point(data=d,aes(x=longitude,y=latitude,fill=mean_LD1,size=n),shape=21,stroke=0.2)+ #can flip LD1 v LD2 here
  guides(fill=guide_colorbar(barheight = unit(20,"mm"),barwidth = unit(4,"mm")),
         size=guide_legend(keyheight = unit(2,"mm")))
dev.off()

#animated gif of (centered) predictions during training
library(gganimate)
a2 <- fread("out/hgdp_chr1_1e6snpsv3_training_preds.txt")
names(a2) <- c("LD1","LD2","sampleID","epoch")
b2 <- fread("data/hgdp/hgdp_sample_data.txt")
c2 <- merge(a2,b2,by="sampleID")
c2 <- ddply(c2,.(epoch),function(k) { #center predictions for prettier plots
  k$LD1 <- k$LD1-mean(k$LD1)
  k$LD2 <- k$LD2-mean(k$LD2)
  return(k)
})
ranges <- ddply(c2,.(epoch),summarize,r2=max(LD2)-min(LD2),r1=max(LD1)-min(LD1))
dropranges <- subset(ranges,r2==max(ranges$r2)|r1==max(ranges$r1))
c2 <- subset(c2,!(epoch %in% dropranges$epoch))
ggplot(c2,aes(x=-LD2,y=-LD1,fill=region))+
  geom_point(shape=21,stroke=0.2,size=3)+
  scale_fill_brewer(palette = "Paired")+
  transition_manual(epoch)+
  guides(fill=guide_legend(override.aes = list(size=4)))+
  labs(title='Epoch: {frame}')
anim_save("out/hgdp_chr1_1e6snpsv3_training_preds.gif")

###############################################
#################### pabu #####################
###############################################
a <- fread("out/pabu_latent_coords.txt",header=T)
names(a) <- c("LD1","LD2","sampleID")
pc <- fread("out/pabu_pca.txt")[,c(1,2,21)]
names(pc) <- c("LD1","LD2","sampleID")
b <- fread("data/pabu/pabu_full_data.txt")
c <- merge(a,b,by="sampleID")
d <- merge(b,pc,by="sampleID")
c$method <- "VAE"
c$k3clust <- kmeans(c[,c("LD1","LD2")],centers=3,iter.max = 10000)[[1]]
d$method <- "PCA"
d$k3clust <- kmeans(d[,c("LD1","LD2")],centers=3,iter.max = 10000)[[1]]
e <- rbind(c,d)

#final prediction
pdf("out/pabu_vae_v_pca.pdf",useDingbats = F,width=6,height=2.5)
ggplot(e,aes(x=-LD1,y=-LD2,fill=factor(k3clust)))+
  facet_wrap(~method,scales="free")+
  geom_point(shape=21,stroke=0.2)+
  scale_fill_brewer(palette = "Dark2")
dev.off()

pdf("out/pabu_map.pdf",useDingbats = F,width=5,height=3)
load("~/locator/locator_hgdp/cntrymap.Rdata")
map <- crop(map,c(-120,-60,10,43))
states <- map_data("state")
f <- ddply(c,.(locality.lat,locality.long,k3clust),summarize,n=length(YR))
f$k3clust <- factor(f$k3clust,levels=levels(factor(e$k3clust)))
ggplot()+coord_map()+
  xlim(-120,-60)+ylim(10,43)+
  geom_polygon(data=map,aes(x=long,y=lat,group=group),lwd=0.2,col="white",fill="grey80")+
  geom_path(data=states,aes(x=long,y=lat,group=group),col="white",lwd=0.2,linetype=2)+
  geom_point(data=f,aes(x=locality.long,y=locality.lat,fill=factor(k3clust),size=n),shape=21,stroke=0.2)+
  scale_fill_brewer(palette="Dark2")
dev.off()

#
