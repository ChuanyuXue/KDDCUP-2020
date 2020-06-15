from modules import *


class Model:
    def __init__(self, usernum, itemnum, args, emb=None, num_neg=2, dec_step=None, 
                 emb_usr=None, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen, num_neg))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=False,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            
            
            # self.lookup_table2 = tf.get_variable('lookup_table2',
            #                   dtype=tf.float32,
            #                   shape=[itemnum + 1, args.hidden_units],
            #                   trainable=False
            #                   )
            # item_emb_table = lookup_table2 + item_emb_table
#            
#            self.seq = tf.nn.embedding_lookup(item_emb_table, self.input_seq)
            
            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            
            # user embedding
            u_, user_emb_table = embedding(
                self.u,
                vocab_size=usernum+1,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="user_embedding",
                reuse=reuse,
                with_t=True
            )
            
            self.seq += t
            
#            user_emb = tf.reshape(u_, [tf.shape(self.input_seq)[0], 1, args.hidden_units])
#            self.seq = user_emb + self.seq
            
            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    
                    self.seq *= mask

            self.seq = normalize(self.seq)
#        print(item_emb_table.shape)
#        print(emb_item.shape)  
        self.emb_item = tf.Variable(emb, dtype=tf.float32)
        self.usr_emb = tf.Variable(emb_usr, dtype=tf.float32)
        
        self.item_emb_table = item_emb_table
#        self.lookup_table2 = lookup_table2
        
        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
#        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen * num_neg])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        
        # ------------------
        #user emedding
        self.user_emb_table = user_emb_table
        user_emb = tf.nn.embedding_lookup(self.user_emb_table, self.u)
        user_emb = tf.reshape(user_emb, [tf.shape(self.input_seq)[0], 1, args.hidden_units])
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0], args.maxlen, args.hidden_units])
        self.seq = user_emb + seq_emb
               
        # last 5 emb
#        item_emb2 = tf.nn.embedding_lookup(item_emb_table, self.input_seq)
#        item_emb2 = tf.reshape(item_emb2, [tf.shape(self.input_seq)[0], args.maxlen, args.hidden_units])
#        item_emb2 = tf.layers.dense(item_emb2, args.hidden_units, activation=None)
#        self.seq = self.seq + item_emb2
        
#        seq_emb = tf.reshape(seq_emb, [-1, args.hidden_units])
#        item_emb2 = tf.reduce_mean(item_emb2[:,-10:,:], axis=1)
        
        # -----------
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

  
        self.test_item = tf.placeholder(tf.int32, shape=(None))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, -1])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        
#        print(neg_emb.shape)
        tmp_seq_emb = tf.reshape(seq_emb, [-1,1,args.hidden_units])
        neg_emb = tf.reshape(neg_emb, [-1,num_neg, args.hidden_units])
        self.neg_logits = tf.reduce_sum(neg_emb * tmp_seq_emb, -1)
        
        self.neg_logits = tf.reshape(self.neg_logits, [tf.shape(self.input_seq)[0] * args.maxlen, num_neg])
        # ignore padding items (0)
        
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        
        # self.pos_logits = tf.reshape(self.pos_logits, [tf.shape(self.input_seq)[0] * args.maxlen, 1])
        # err = self.pos_logits - self.neg_logits 
        # self.loss = tf.reduce_sum(
        #     -tf.reduce_sum(tf.log(tf.sigmoid(err) + 1e-24), axis=-1) * istarget
        # ) / tf.reduce_sum(istarget)
        
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.reduce_sum(tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24), axis=-1) * istarget
        ) / tf.reduce_sum(istarget)
        
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.lr = tf.train.exponential_decay(args.lr,
                                self.global_step, dec_step, 0.5, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta2=0.98)

            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})