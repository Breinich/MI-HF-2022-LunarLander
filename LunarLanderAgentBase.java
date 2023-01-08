public class LunarLanderAgentBase {
    // The resolution of the observation space
    // The four variables of the observation space, from left to right:
    //   0: X component of the vector pointing to the middle of the platform from the lander
    //   1: Y component of the vector pointing to the middle of the platform from the lander
    //   2: X component of the velocity vector of the lander
    //   3: Y component of the velocity vector of the lander
    static final int[] OBSERVATION_SPACE_RESOLUTION = {7, 9, 14, 14};

    final double[][] observationSpace;
    double[][][][][] qTable;
    final int[] envActionSpace;
    private final int nIterations;

    double epsilon = 1.0;
    int iteration = 0;
    boolean test = false;

    // your variables here
    // ...
    double[][][][][] bestTable;
    double bestReward = -200;
    double lastReward = -200;

    double alpha = 0.01;
    double gamma = 0.83;
    int epsilon_step = 100;
    double epsilon_decay = 0.91;
    int save_interval = 1000;

    int epoch = 0;

    public LunarLanderAgentBase(double[][] observationSpace, int[] actionSpace, int nIterations) {
        this.observationSpace = observationSpace;
        this.qTable =
                new double[OBSERVATION_SPACE_RESOLUTION[0]]
                        [OBSERVATION_SPACE_RESOLUTION[1]]
                        [OBSERVATION_SPACE_RESOLUTION[2]]
                        [OBSERVATION_SPACE_RESOLUTION[3]]
                        [actionSpace.length];
        this.envActionSpace = actionSpace;
        this.nIterations = nIterations;
    }

    /**
     * Visszaadja a kapott folytonos allapothoz tartozo kvantalt erteket.
     * @param observationSpace
     * @param state
     * @return
     */
    public static int[] quantizeState(double[][] observationSpace, double[] state) {
        return new int[]{(int) ((state[0]-observationSpace[0][0]) / ((observationSpace[0][1]-observationSpace[0][0])/(OBSERVATION_SPACE_RESOLUTION[0]-1))),
                (int) ((state[1]-observationSpace[1][0]) / ((observationSpace[1][1]-observationSpace[1][0])/(OBSERVATION_SPACE_RESOLUTION[1]-1))),
                (int) ((state[2]-observationSpace[2][0]) / ((observationSpace[2][1]-observationSpace[2][0])/(OBSERVATION_SPACE_RESOLUTION[2]-1))),
                (int) ((state[3]-observationSpace[3][0]) / ((observationSpace[3][1]-observationSpace[3][0])/(OBSERVATION_SPACE_RESOLUTION[3]-1)))
        };
    }

    public void epochEnd(double epochRewardSum) {
        epsilon *= epsilon_decay;
    }

    private double getMaxValue(double[] array) {
        double maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
            }
        }
        return maxValue;
    }

    public void learn(double[] oldState, int action, double[] newState, double reward) {
        //System.out.println(reward);
        int[] OldState = quantizeState(observationSpace, oldState);
        int[] NewState = quantizeState(observationSpace, newState);

        qTable[OldState[0]][OldState[1]][OldState[2]][OldState[3]][action] =
                qTable[OldState[0]][OldState[1]][OldState[2]][OldState[3]][action] +
                        alpha*(reward + gamma*(getMaxValue(qTable[NewState[0]][NewState[1]][NewState[2]][NewState[3]])) -
                                qTable[OldState[0]][OldState[1]][OldState[2]][OldState[3]][action]);
    }

    public void trainEnd() {
        test = false;
    }
}
