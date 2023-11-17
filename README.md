# Data Battle Kemenkeu RI - Penerimaan Negara II - Manajemen Risiko Layanan Kepabeanan

<p align="center">
  <img src="https://3.bp.blogspot.com/-AuZmxe77T_k/WssHJWNtEjI/AAAAAAAAVPw/k13CLs_MXy8VdPs8iDWZjbLr2EncWgdWQCLcBGAs/w1200-h630-p-k-no-nu/kemenkeu.png" alt="Kemenkeu Logo">
</p>

## Konteks Kompetisi
Konteks kompetisi ini berkisar pada manajemen risiko dalam layanan kepabeanan guna mengamankan penerimaan negara dari sektor bea dan cukai. Manajemen risiko yang efisien sangat penting untuk penyelenggaraan layanan kepabeanan. Pembagian barang impor menjadi merah untuk risiko tinggi hingga hijau untuk risiko rendah diyakini dapat meningkatkan efektivitas dan efisiensi proses pemeriksaan barang yang masuk ke dalam daerah pabean. Kemampuan yang memadai dalam mengelompokkan dan/atau mengklasifikasikan transaksi impor ke dalam kategori risiko sangat diperlukan dalam proses bisnis ini. Kegagalan dalam menerapkan metode yang tepat untuk mengelompokkan dan/atau mengklasifikasikan transaksi impor dapat berdampak pada rendahnya tingkat efektivitas dan efisiensi proses pemeriksaan barang. Pada kasus ini, peserta diminta untuk mengembangkan model klasifikasi berdasarkan data historis yang diberikan, kemudian menerapkan model yang telah dilatih untuk mengklasifikasikan dataset yang telah diberikan. Penilaian dilakukan dengan mengukur tingkat akurasi hasil prediksi dibandingkan dengan nilai aktual. Pemenang dipilih dari peserta dengan nilai prediksi yang memiliki tingkat akurasi paling tinggi.

## Fitur Data

| Field                | Definisi                                                   |
|----------------------|------------------------------------------------------------|
| DocsDate             | Tanggal Dokumen Impor                                      |
| HSCODE               | Kode Sistem Harmonis, 4 digit. [Definisi Umum](https://en.wikipedia.org/wiki/Harmonized_System), [Pengenalan](https://klc2.kemenkeu.go.id/kms/knowledge/klc1-klasifikasi-kepabeanan-dan-cukai/detail/)|
| CountryOfOrigin      | Negara Asal Barang                                         |
| Using_Intermediaries | Apakah kegiatan impor dilakukan secara mandiri oleh Wajib Pajak atau dengan bantuan pihak ketiga (PPJK). 1 = Menggunakan Intermediaries, 0 = Tidak menggunakan Intermediaries |
| Using_TradeAgreement  | Apakah kegiatan impor memanfaatkan fasilitas perjanjian perdagangan antar negara. 1 = Menggunakan Trade Agreement, 0 = Tidak menggunakan Trade Agreement |
| CustChannel          | Jalur Impor dari DJBC (Direktorat Jenderal Bea dan Cukai)   |
| ImportDuty           | Penerimaan Pajak dari kegiatan Impor                        |

## Pendekatan yang saya lakukan
Disini pada awalnya saya sudah mencoba untuk melakukan modeling dengan xgboost+hyperparameter optuna, tetapi disini pendekatan yang digunakan masih sangatlah sederhana dengan hanya melakukan preprocessing seperti encode fitur kategorik tanpa mencoba menggali insight untuk menambah fitur baru (as simple as that 😁). The result?
Well, ini sudah bagus untuk menjadi baseline dimana didapatkan hasil public score sbg berikut: `0.90982`. More surprising when see that the private score lebih bagus, dengan acc -> `0.91038`, But yah dengan score acc seperti itu, di leaderboard saya masih terseok seok dengan rank 30an 😓, at the end of competition with score like that we can only ranked 50th++ hehe.

### Feature Engineering
Setelah membaca baca beberapa case tentang ekonomi dsb, saya mendapatkan ide untuk mencreate beberapa fitur baru berdasarkan dari fitur yang telah disediakan oleh panitia penyelenggara. Adapun workflow yang saya lakukan adalah sbg berikut. 

[<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=15xlf1wFZa1EAHFmdxDlSLBw8bPxq3rj4" alt="Feature Engineering">
</p>

**Pertanyaanya** Apakah dengan fitur seperti itu dan modeling yang sama score yang didapat increase? Oh tentu saja, disini nilai acc naik di angka `0.92579`. Perlahan rank juga naik ke 20an (😊senyum bangga wkwk), di acc score 0.92 ini saya belum menambah `fitur engineering untuk HSCODE dengan mengenerate HSCODE_Category dan Industry_Category`. Setelah melakukan beberapa trial n error menggunakan beberapa teknik ensembling, voting dll didapat stacking menjadi pendekatan terbaik dimana disaat melakukan stacking hanya dengan model baseline XGBOOST dan LGBM nilai acc juga meningkat dengan score `0.94262`. Akhirnya setelah memutuskan untuk melakukan modeling dengan pendekatan stacking, saya mencoba untuk melakukan hyperparameter pada setiap algoritma baik XGBOOST, LGBM dan meta classifier Logit. Hasilnya? Alhamdulillah lagi lagi memuaskan disini dengan score `0.94836` (yah altough naiknya ga signifikan). <br>
[IDEA HSCODE]<br>
Hari berganti hari.... 🧨DUARRR, disini saya mencoba menggali informasi lebih lanjut dengan menonton video course dari KEMENKEU learning center-PKN STAN, dan mendapatkan ide dari klasifikasi HSCODE dari wikipedia -> https://en.wikipedia.org/wiki/Harmonized_System. Saya mencoba untuk submit...dannnn Alhamdulillah lagi lagi score increase di `0.95296`
, btw ini h-2 finish compe (honestly saya udah seneng dengan hasil tsb sampe saya ss wkwkwk)

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=19v2GqCSUZ26DsRCKQnDptyAjUQ2HXrNY" alt="Rank 14">
</p>

Dihari ini juga saya mencoba untuk mencoba tujuh baseline model berbasis tree dan boosting, mulai dari ExtraTree, GBT, AdaBoost, LightGBM, Decision Tree, Random Forest, serta HistGradBoost. Disini ternyata 4 model teratas yang didapat adalah ExtraTree, RandomForest, Xgboost, LGBM. So, dengan trial tsb saya memutuskan untuk menambah model stacking saya dengan 2 algoritma teratas yaitu ExtraTree dan RandomForest. Hasilnya? skarang ga naik wkwkwk 😂, nilai masih sama di angka `0.95an`

### Feature Selection
Last Day....Hari ini sebenarnya lebih ke eksplorasi lebih lanjut saja mengenai data karena untuk modeling saya rasa sudah stuck disitu situ saja, kalaupun increase tidak bisa signifikan. Disini saya memutuskan untuk tidak menggunakan seluruh fitur ke fase modeling, hal yang dilakukan tentu saja melakuka feature selection, disini saya ada mencoba menggunakan beberapa teknik Wrapper seperti Recursive Feature Elimination serta filtering dengan corr analysis serta Information Gain (Entropy), pada akhirnya disaat melakukan trial n error pendekatan terbaik jatuh ke teknik  SHAP (SHapley Additive exPlanations), untuk visualisasinya seperti berikut.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1Bf-dF1IZBUCCY8zt916-0R1THl4pNhiJ" alt="Shap">
</p>

Dimana 8 fitur terpenting disini `Using_intermediaries, HSCODE, CountryOfOrigin, Valuta_CIF, Trade_duration, CIF, Category serta Day`. Setelah proses model selection tsb, Alhamdulillah finally score increase lagi kali ini di angka `0.96693`. Di leaderboard juga merengsek naik ke top 10.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1Rnc6Ps8shycDiyQsNhM8tgzfvf6RSZjj" alt="top10">
</p>

Oiya untuk penjelasan lebih lanjut mengenai `shap` bisa dibaca disini:
- https://www.kaggle.com/code/diegovicente/using-shap-values-for-interpretability
- https://shap.readthedocs.io/en/latest/
- https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137


### Modeling
### Model Evaluation




