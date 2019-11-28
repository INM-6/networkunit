
import brunel_model

def test_brunel_model():

    bm = brunel_model.brunel_model(name='pytest')
    bm.run()

    pass


if __name__ == '__main__':
    # XXX This is a shitty testing env, only used for development
    test_brunel_model()
    print('\n ------------------------------------\n')
    print('All tests pass!\n\n')
