"""
IMPORTANT
1. milvus operate
2. load embedding_table into milvus
"""
import re
import time
import pyspark.sql.functions as F


from pymilvus import CollectionSchema, FieldSchema, DataType, connections, Collection, utility

# EMBEDDING_TABLE = "test.sim_matrix_vector_tf_posneg_array_2"
EMBEDDING_TABLE = "test.sim_matrix_embedding_nce"

YINGQU_DETAILPAGE_SEARCH_PREFIX = "yingqu_detailpage_search_"
YINGQU_DETAILPAGE_SEARCH_ALIAS = re.sub("_$", "", YINGQU_DETAILPAGE_SEARCH_PREFIX)
YINGQU_DETAILPAGE_QUERY_PREFIX = "yingqu_detailpage_query_"
YINGQU_DETAILPAGE_QUERY_ALIAS = re.sub("_$", "", YINGQU_DETAILPAGE_QUERY_PREFIX)

YINGQU_HOMEPAGE_QUERY_PREFIX = "yingqu_homepage_query_"
YINGQU_HOMEPAGE_SEARCH_PREFIX = "yingqu_homepage_search_"

PROJECT = 1

connections.connect(alias="default", host='10.111.10.98', port='19531')


def createCollection(prefix, t):
    video_id = FieldSchema(name="video_id", dtype=DataType.INT64, is_primary=True, auto_id=False)

    embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=32)

    vip_status = FieldSchema(name="vip_status", dtype=DataType.INT64)

    needbuy_status = FieldSchema(name="needbuy_status", dtype=DataType.INT64)

    schema = CollectionSchema(
        fields=[video_id, embedding, vip_status, needbuy_status],
        description="None"
    )
    collection_name = prefix + t
    collection = Collection(
        name=collection_name,
        schema=schema,
        using='default',
        shards_num=4
    )
    return collection


def get_data():
    df = spark.sql(f"select video_id, embedding_query, embedding_search  from {EMBEDDING_TABLE}")

    itemInfoDF = spark.sql(f"""select cast(course_id as string) course_id, is_vip vip_status, is_needbuy needbuy_status, permit_client 
            from (select course_id, permit_client from ods.commondb_course_platform where platform = {PROJECT} and status = 1 group by course_id, permit_client) a
            inner join
            ( select id, is_vip, is_needbuy from ods.commondb_course ) b on a.course_id = b.id """).select("course_id",
                                                                                                           "vip_status",
                                                                                                           "needbuy_status")

    data = df.join(itemInfoDF, df.video_id == itemInfoDF.course_id, "left") \
        .withColumn("vip_status", F.coalesce(F.col("vip_status"), F.lit(0))) \
        .withColumn("needbuy_status", F.coalesce(F.col("needbuy_status"), F.lit(0))) \
        .select(F.col("video_id").cast("Long"), F.col("embedding_query").cast("array<float>"),
                F.col("embedding_search").cast("array<float>"), F.col("vip_status").cast("long"),
                F.col("needbuy_status").cast("long")).collect()

    video_id = [i[0] for i in data]
    embedding_query = [i[1] for i in data]
    embedding_search = [i[2] for i in data]
    vip_status = [i[3] for i in data]
    needbuy_status = [i[4] for i in data]
    return video_id, embedding_query, embedding_search, vip_status, needbuy_status


time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
collection_query = createCollection(YINGQU_DETAILPAGE_QUERY_PREFIX, time_stamp)
collection_search = createCollection(YINGQU_DETAILPAGE_SEARCH_PREFIX, time_stamp)

video_id, embedding_query, embedding_search, vip_status, needbuy_status = get_data()
tmp_slice = None
collection_query.insert(
    [video_id[0:tmp_slice], embedding_query[0:tmp_slice], vip_status[0:tmp_slice], needbuy_status[0:tmp_slice]])
collection_search.insert(
    [video_id[0:tmp_slice], embedding_search[0:tmp_slice], vip_status[0:tmp_slice], needbuy_status[0:tmp_slice]])

# index process  nlist = 1024 probe = 32
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}

collection_search.create_index(
    field_name="embedding",
    index_params=index_params
)

collection_search.create_index(
    field_name="vip_status",
    index_name="vip_status_index",
)

collection_search.create_index(
    field_name="needbuy_status",
    index_name="needbuy_status_index",
)

collection_search.create_index(
    field_name="video_id",
    index_name="video_id_index",
)

collection_query.create_index(
    field_name="embedding",
    index_params=index_params
)
collection_query.create_index(
    field_name="video_id",
    index_name="video_id_index",
)

collection_search.load()
collection_query.load()

collection_query.flush()
collection_search.flush()

utility.alter_alias(
    collection_name=YINGQU_DETAILPAGE_QUERY_PREFIX + time_stamp,
    alias=YINGQU_DETAILPAGE_QUERY_ALIAS
)
utility.alter_alias(
    collection_name=YINGQU_DETAILPAGE_SEARCH_PREFIX + time_stamp,
    alias=YINGQU_DETAILPAGE_SEARCH_ALIAS
)

utility.list_collections

print(Collection(YINGQU_DETAILPAGE_QUERY_ALIAS).num_entities, Collection(YINGQU_DETAILPAGE_SEARCH_ALIAS).num_entities)

query_collections = sorted(
    list(filter(lambda s: re.match(YINGQU_DETAILPAGE_QUERY_PREFIX, s) != None, utility.list_collections())))

search_collections = sorted(
    list(filter(lambda s: re.match(YINGQU_DETAILPAGE_SEARCH_PREFIX, s) != None, utility.list_collections())))

for i in query_collections[1:-3]:
    print("delete = ", i)
    utility.drop_collection(i)

for i in search_collections[1:-3]:
    print("delete = ", i)
    utility.drop_collection(i)


search_param = {
  "data": vec,
  "anns_field": "embedding",
  "param": {"metric_type": "IP", "params": {"nprobe": 32}},
  "offset": 4,
  "limit": 1024,
  "expr": "vip_status == 0 and needbuy_status == 0"
  #"expr": None
}

results = collection_search.search(**search_param)
