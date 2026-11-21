from pathlib import Path

# Proje ana dizini (bilgisayardan bağımsız göreceli yol)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Deney tekrarlanabilirliği için sabit tohum
RANDOM_STATE = 42

# Hedef ve hata analizi meta sütunları
TARGET_COL = "Attack_label"
META_COL = "Attack_type"

# Temizlemede çıkarılacak 27 sütun (sızıntı + gürültü + sıfır varyans)
DROP_COLS = [
    # 1. Zaman ve kimlik sızıntıları (6)
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "icmp.transmit_timestamp",
    # 2. Ham payload ve yüksek kardinaliteli testbed metinleri (9)
    "tcp.payload",
    "tcp.options",
    "http.file_data",
    "http.request.full_uri",
    "http.request.uri.query",
    "http.referer",
    "dns.qry.name",
    "mqtt.msg",
    "mqtt.topic",
    # 3. Rastgele ID, checksum ve Wireshark akış indeksi (5)
    "icmp.checksum",
    "tcp.checksum",
    "tcp.ack_raw",
    "mbtcp.trans_id",
    "udp.stream",
    # 4. EDA'da tespit edilen sıfır varyanslı sabit sütunlar (7)
    "icmp.unused",
    "http.tls_port",
    "dns.qry.type",
    "dns.retransmit_request_in",
    "mqtt.msg_decoded_as",
    "mbtcp.len",
    "mbtcp.unit_id",
]

# Format temizliği sonrası geriye kalan tek 2 kategorik sütun
CATEGORICAL_COLS = [
    "http.request.method",
    "http.request.version",
]