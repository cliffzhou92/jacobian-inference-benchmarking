import examples

def main():
    # double positive monostable switch
    fp, B = examples.double_positive()
    examples.sensitivity(fp[0])

    # double negative bistable switch
    fp, B = examples.double_negative()
    examples.sensitivity(fp[0], tag='double_neg_')

    # repressilator
    fp, B = examples.repressilator()
    examples.sensitivity(fp[0], tag='repressilator_')

    # tristable switch
    fp, B = examples.tristable()
    examples.sensitivity(fp[2], tag='tristable_')


if __name__=='__main__':
    main()
