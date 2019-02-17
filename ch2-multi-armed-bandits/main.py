import tensorflow as tf
import argparse


def create_graph(params):
    tf.reset_default_graph()

    arms_count = params.arms
    games_count = params.games
    shape = (games_count, arms_count)

    q = tf.get_variable(
            'q',
            shape,
            initializer=tf.initializers.random_normal)
    q_estimated = tf.get_variable(
            'q_estimated',
            shape,
            initializer=tf.zeros_initializer)
    n = tf.get_variable(
            'n',
            shape,
            dtype=tf.int32,
            initializer=tf.ones_initializer)

    with tf.variable_scope('action_selection'):
        greedy_action = tf.argmax(
                q_estimated,
                axis=1,
                name='greedy_action',
                output_type=tf.int32)
        random_action = tf.random_uniform(
                shape=(games_count,),
                maxval=arms_count,
                dtype=tf.int32,
                name='random_action')
        epsilon = tf.constant(params.epsilon, name='epsilon')
        selected_action = tf.where(
                tf.random_uniform(shape=(games_count,), maxval=1.0) < epsilon,
                random_action,
                greedy_action,
                name='selected_action')

    mask = tf.one_hot(selected_action,
                      depth=arms_count,
                      dtype=tf.float32,
                      name='mask')

    with tf.variable_scope('reward'):
        reward = (tf.random_normal(shape) + q) * mask

    with tf.variable_scope('update_q'):
        error = reward - q_estimated * mask
        inc_n = tf.assign_add(
                n,
                tf.cast(mask, tf.int32),
                name='increment_n')
        delta = error / tf.cast(inc_n, tf.float32)
        train_op = tf.assign_add(
                q_estimated,
                delta,
                name='update_q_estimated')

    with tf.variable_scope('stat'):
        # Optimal action ratio
        optimal_action = tf.argmax(
                q,
                axis=1,
                output_type=tf.int32,
                name='optimal_action')
        is_optimal_action = tf.equal(selected_action, optimal_action)
        optimal_action_ratio = tf.reduce_mean(
                tf.cast(is_optimal_action, tf.float32))
        tf.summary.scalar('optimal_action_ratio', optimal_action_ratio)

        # Average reward
        avg_reward = tf.reduce_mean(tf.reduce_sum(reward, axis=1))
        tf.summary.scalar('avg_reward', avg_reward)

        # True error
        true_error = tf.reduce_sum((q_estimated - q) * mask, axis=1)
        tf.summary.histogram('true_error', true_error)
        tf.summary.scalar('avg_true_error', tf.reduce_mean(true_error))

        tf.summary.histogram('reward', tf.reduce_sum(reward, axis=1))
        tf.summary.histogram('error', tf.reduce_sum(error, axis=1))
        tf.summary.histogram('delta', tf.reduce_sum(delta, axis=1))
        tf.summary.histogram('q_estimated', q_estimated)

        summaries = tf.summary.merge_all()

    return summaries, train_op


def run_experiment(params):

    summaries, train_op = create_graph(params)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(params.output_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        for step in range(params.steps):
            summ, _ = sess.run([summaries, train_op])
            writer.add_summary(summ, global_step=step)

        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Run multi-armed bandit experiment')
    parser.add_argument('--epsilon',
                        type=float,
                        help='Ratio of exploration actions')
    parser.add_argument('--games',
                        type=int,
                        default=2000,
                        help='Number of games')
    parser.add_argument('--steps',
                        type=int,
                        default=1000,
                        help='Number of steps for each game')
    parser.add_argument('--arms',
                        type=int,
                        default=10,
                        help='Number of bandit arms')
    parser.add_argument('output_dir', type=str,
                        metavar='output_dir',
                        help='Statistics directory for Tensorboard')
    args = parser.parse_args()
    run_experiment(args)
