model.py

##### 1st: Encoding

# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

# Classification block
# x = Flatten(name='flatten')(x)
# x = Dense(4096, activation='relu', name='fc1')(x)
# # dropout
# x = Dense(4096, activation='relu', name='fc2')(x)
# # dropout
# x = Dense(classes, activation='softmax', name='predictions')(x)

## change 2 flatten layers by 1x1 conv
x = Conv2d(512, (1,1), activation='relu', name='fc1')(x)
# dropout 0.5
x = Conv2d(512, (1,1), activation='relu', name='fc1')(x)
# dropout 0.5


##### 2nd: Decoding

# 1x1 convolution
n.score_fr_sem = L.Convolution(n.drop7, num_output=33, kernel_size=1, pad=0,
    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

# deconvolution with stride 2, upsample
n.upscore2_sem = L.Deconvolution(n.score_fr_sem,
    convolution_param=dict(num_output=33, kernel_size=4, stride=2,
        bias_term=False),
    param=[dict(lr_mult=0)])


# 1x1 convolution
n.score_pool4_sem = L.Convolution(n.pool4, num_output=33, kernel_size=1, pad=0,
    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

# not sure
n.score_pool4_semc = crop(n.score_pool4_sem, n.upscore2_sem)

# fuse with VGG layer
n.fuse_pool4_sem = L.Eltwise(n.upscore2_sem, n.score_pool4_semc,
        operation=P.Eltwise.SUM)

# deconvolution with stride 2, upsample
n.upscore_pool4_sem  = L.Deconvolution(n.fuse_pool4_sem,
    convolution_param=dict(num_output=33, kernel_size=4, stride=2,
        bias_term=False),
    param=[dict(lr_mult=0)])

# 1x1 convolution
n.score_pool3_sem = L.Convolution(n.pool3, num_output=33, kernel_size=1,
        pad=0, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2,
            decay_mult=0)])

# not sure
n.score_pool3_semc = crop(n.score_pool3_sem, n.upscore_pool4_sem)

# fuse with VGG layer
n.fuse_pool3_sem = L.Eltwise(n.upscore_pool4_sem, n.score_pool3_semc,
        operation=P.Eltwise.SUM)

# deconvolution with stride 8, upsample
n.upscore8_sem = L.Deconvolution(n.fuse_pool3_sem,
    convolution_param=dict(num_output=33, kernel_size=16, stride=8,
        bias_term=False),
    param=[dict(lr_mult=0)])

# not sure
n.score_sem = crop(n.upscore8_sem, n.data)

# loss to make score happy (o.w. loss_sem)
n.loss = L.SoftmaxWithLoss(n.score_sem, n.sem,
        loss_param=dict(normalize=False, ignore_label=255))

# 1x1 convolution
n.score_fr_geo = L.Convolution(n.drop7, num_output=3, kernel_size=1, pad=0,
    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

# decon
n.upscore2_geo = L.Deconvolution(n.score_fr_geo,
    convolution_param=dict(num_output=3, kernel_size=4, stride=2,
        bias_term=False),
    param=[dict(lr_mult=0)])

n.score_pool4_geo = L.Convolution(n.pool4, num_output=3, kernel_size=1, pad=0,
    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
n.score_pool4_geoc = crop(n.score_pool4_geo, n.upscore2_geo)
n.fuse_pool4_geo = L.Eltwise(n.upscore2_geo, n.score_pool4_geoc,
        operation=P.Eltwise.SUM)
n.upscore_pool4_geo  = L.Deconvolution(n.fuse_pool4_geo,
    convolution_param=dict(num_output=3, kernel_size=4, stride=2,
        bias_term=False),
    param=[dict(lr_mult=0)])

n.score_pool3_geo = L.Convolution(n.pool3, num_output=3, kernel_size=1,
        pad=0, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2,
            decay_mult=0)])
n.score_pool3_geoc = crop(n.score_pool3_geo, n.upscore_pool4_geo)
n.fuse_pool3_geo = L.Eltwise(n.upscore_pool4_geo, n.score_pool3_geoc,
        operation=P.Eltwise.SUM)
n.upscore8_geo = L.Deconvolution(n.fuse_pool3_geo,
    convolution_param=dict(num_output=3, kernel_size=16, stride=8,
        bias_term=False),
    param=[dict(lr_mult=0)])

n.score_geo = crop(n.upscore8_geo, n.data)
n.loss_geo = L.SoftmaxWithLoss(n.score_geo, n.geo,
        loss_param=dict(normalize=False, ignore_label=255))

return n.to_proto()

    # 3rd block
    x = tf.layers.conv2d(input=x, filters=256, kernel_size=(3,3), 
        activation=tf.nn.relu)
    x = tf.layers.conv2d(input=x, filters=256, kernel_size=(3,3), 
        activation=tf.nn.relu)
    x = tf.layers.conv2d(input=x, filters=256, kernel_size=(3,3), 
        activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(input=x, pool_size=(2,2), strides=(2,2))


    # 4th block
    x = tf.layers.conv2d(input=x, filters=512, kernel_size=(3,3), 
        activation=tf.nn.relu)
    x = tf.layers.conv2d(input=x, filters=512, kernel_size=(3,3), 
        activation=tf.nn.relu)
    x = tf.layers.conv2d(input=x, filters=512, kernel_size=(3,3), 
        activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(input=x, pool_size=(2,2), strides=(2,2))


    # 5th block
    x = tf.layers.conv2d(input=x, filters=512, kernel_size=(3,3), 
        activation=tf.nn.relu)
    x = tf.layers.conv2d(input=x, filters=512, kernel_size=(3,3), 
        activation=tf.nn.relu)
    x = tf.layers.conv2d(input=x, filters=512, kernel_size=(3,3), 
        activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(input=x, pool_size=(2,2), strides=(2,2))

