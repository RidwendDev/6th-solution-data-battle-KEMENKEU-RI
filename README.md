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
### Feature Engineering
<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=15xlf1wFZa1EAHFmdxDlSLBw8bPxq3rj4" alt="Feature Engineering">
</p>

### Feature Selection
### Modeling
### Model Evaluation




