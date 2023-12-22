import pickle


SERIALIZED_STEP = "<serialized-step>"


if __name__ == '__main__':
    step = pickle.loads(SERIALIZED_STEP)
    step.execute()
