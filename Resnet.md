## Gösterilecekler
 - [ ] 2 Çeşit resnet bloğu var
	- Normal
	- Bottleneck
 - [x] Yazarlar hakkında kısa bilgi
 - [ ] train durumda loss test durumda loss ile overfit olmadığını göster
 - [ ] Imagenet dataset
 - [ ] Degradation problem
 - [ ] Overfit nedir
	 - [ ] Degradationdan farkı nedir
 - [ ] VGG modeli
 - [ ] COCO object detection dataset
 - [ ] Is learning better networks as easy as stacking more layers?
 - [ ] Degradation problem
 - [ ] **Şu cümle**  _Let us consider a shallower architecture and its deeper counterpart that adds more layers onto it. There exists a solution by construction to the deeper model: the added layers are identity mapping, and the other layers are copied from the learned shallower model. The existence of this constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart_ 
 - [ ] ILSVRC 2015 classification competition
 - [ ] Fazla olan layerlerin identity mapping yaklaşması (mümküne görselleştir)
 - [ ] Doğrudan bir fonksiyon öğrenmek yerine "zaten iyi çözüm" + "küçük düzeltme" öğreniyor

## Slaytlar
*  **slayt 1** 
	* Merhaba Sizlere Residual network for image classification makalesini sunacağım. 
	* Bu makale 2015'te Microsoft research asia tarafından yayınlanmış olup 2016 yılında cvpr'da **Best Paper Award** almıştır.
* **slayt 2**
	* Makalenin çözüm sunduğu sorunu anlamak için öncelikle bazı terimleri bilmemiz gerekiyor. Bunlardan biri **Derin ağlar**
	* Derin sinir ağlar 1 den fazla gizli layer içeren sinir ağlarıdır.
	* Ağlarda derinlik layer sayısı ile belirlenir daha çok layere sahip olan bir ağ daha derindir denir.
* **slayt 3**
	* Bu görsel, derin sinir ağlarında katmanlar ilerledikçe öğrenilen özelliklerin nasıl değiştiğini gösterir: ilk katmanlar basit kenar ve renkleri öğrenirken, orta katmanlar dokuları ve şekilleri, daha derin katmanlar ise nesne parçalarını ve anlamlı kavramları temsil eder.


* **slayt 4**
	* Daha derin networkler  daha fazla parametreye sahiptirler bu yüzden dataya daha iyi uyum sağlayabilriler.(fitting)
	* Bu sebeple daha karmaşık görevler için daha derin ağlar tercih edilir. 
	* Örneğin resimdeki $y = \sin(2x)\cdot\cos(0.5x) + 0.4\sin(5x)$ fonksiyonu için 2 layere sahip bir network ile 8 layere sahip bir network performansını görüyorsunuz. 
	* Mavi renk datayı kırmızı renk ise modelin tahminini simgeliyor. 
	* Görüldüğü gibi daha sığ model bazen tam değere gelememişken derin network çok daha iyi uyum sağlamış. 
	* Benzer şekilde alttaki resim ise bir classification modeli için sonuçlar. Spiral bir dataset için 2 layerli ve 8 layeri olan bir model train edilmiş ve daha derin olan model çok daha iyi uyum sağlamış (fitting)
	* Bu sonuçlar makaledeki ana soruyu akla getirir *Is learning better networks as easy as stacking more layers?* 



	* Ancak pratikte bu her zaman böyle değil. 
	* Resimdede gördüğünüz gibi 56 layere sahip ağ hem train sırasında hemde test sırasında 20 layere sahip ağa göre daha yüksek loss üretmiştir.

