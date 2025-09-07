import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "../pretrained" # 事前学習モデルの保存先指定

from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv


load_dotenv()
login(token=os.environ["HUGGINGFACE_TOKEN"]) 

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")
# model = SentenceTransformer("pfnet/plamo-embedding-1b", trust_remote_code=True)

# Run inference with queries and documents
# query = "Which planet is known as the Red Planet?"
# documents = [
#     "Venus is often called Earth's twin because of its similar size and proximity.",
#     "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
#     "Jupiter, the largest planet in our solar system, has a prominent red spot.",
#     "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
# ]

# query = "赤い惑星として知られている惑星はどれですか？"
# documents = [
#     "金星は地球と大きさや位置が似ているため、地球の双子と呼ばれることがあります。",
#     "火星は赤みを帯びた外見から、赤い惑星と呼ばれることが多いです。",
#     "木星は太陽系で最も大きな惑星で、大赤斑と呼ばれる巨大な赤い嵐があります。",
#     "土星は美しい環で有名ですが、赤い惑星と間違われることもあります。"
# ]

query = "日本で桜の名所として有名な場所はどこですか？"
documents = [
    "京都の清水寺は、紅葉や歴史的建造物で人気の観光地です。",
    "奈良公園は鹿と触れ合える場所として有名です。",
    "弘前公園は春になると桜が満開になり、日本有数の花見スポットとして知られています。",
    "富士山は日本を代表する山で、世界文化遺産にも登録されています。"
]

query_embeddings = model.encode_query(query)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (768,) (4, 768)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.3011, 0.6359, 0.4930, 0.4889]])