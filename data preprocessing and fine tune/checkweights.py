import tensorflow as tf
sess=tf.Session()
new_saver=tf.train.import_meta_graph()
all_vars=tf.get_collection() #all_vars=tf.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
print(all_vars)
for v in all_vars:
    v_=sess.run(v)
    print(v_)
