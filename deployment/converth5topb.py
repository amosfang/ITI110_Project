# Convert saved h5 model to pb.
import tensorflow as tf

model = tf.keras.models.load_model('./models/vgg16/1/vgg16_u_net.h5')
export_path = './models/vgg16/1/'

tf.saved_model.save(model, export_path)