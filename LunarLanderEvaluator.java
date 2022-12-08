import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import static java.lang.Math.min;

public class LunarLanderEvaluator {
    public static void main(String[] args) {
        var results = new int[5];
        int max_point_count = 0;
        int runs = 100;

        for (int i = 0; i < runs; i++) {
            var result = run();
            results[0] += result.get(0);
            results[1] += result.get(1);
            results[2] += result.get(2);
            results[3] += result.get(3);
            results[4] += result.get(4);
            if (result.get(5) >= 12) {
                max_point_count++;
            }
        }
        double max_point_rate = (max_point_count / (double) runs) * 100;

        System.out.println("Max points:\n\tCount: " + max_point_count + " / " + runs + " runs\n\tRate: " + max_point_rate + "%");
        System.out.println("Results:");
        for (int i = 0; i < 5; i++) {
            System.out.println("\t" + Result.values()[i].toString() + " : " + results[i]);
        }
    }

    public static ArrayList<Integer> run() {
        int nIterations = (int) 1e6;
        int iteration = 0;

        int[][] randomVelocityRange = {{-1, 1}, {1, 5}};
        Environment env = new Environment(randomVelocityRange);

        LunarLanderAgent agent = new LunarLanderAgent(env.observationSpace,
                env.actionSpace, nIterations);

        while (iteration < nIterations) {
            double[] state = env.reset();

            double epochRewardSum = 0;
            boolean done = false;

            while (!done) {
                int action = agent.step(state);
                Environment.EnvStepDTO envStep = env.step(action);
                done = env.done;
                agent.learn(state, action, envStep.state, envStep.reward);

                state = envStep.state;

                epochRewardSum += envStep.reward;

                iteration += 1;
            }

            agent.epochEnd(epochRewardSum);
        }

        agent.trainEnd();

        //System.out.println(Arrays.deepToString(agent.qTable));
        //System.out.println("Random steps  : " + LunarLanderAgent.randoms);
        //System.out.println("Exploit steps : " + LunarLanderAgent.argmaxes);

        int nTestIterations = 10;
        double rewardSum = 0;

        HashMap<Result, Integer> iterationOutcomes = new HashMap<>();
        for (Result resType : Result.values()) {
            iterationOutcomes.put(resType, 0);
        }

        for (int i = 0; i < nTestIterations; i++) {
            double[] state = env.reset();
            boolean done = false;

            while (!done) {
                int action = agent.step(state);
                Environment.EnvStepDTO envStep = env.step(action);
                state = envStep.state;
                done = envStep.done;

                rewardSum += envStep.reward;
            }

            iterationOutcomes.put(env.result, iterationOutcomes.get(env.result) + 1);
        }

        HashMap<Result, Integer> pointDict = new HashMap<>() {{
            put(Result.LANDED, 4);
            put(Result.LANDING_GEAR_CRASHED, 2);
            put(Result.CRASH_LANDING, 1);
            put(Result.CRASH, 0);
            put(Result.OUT_OF_TIME, 0);
        }};

        int maxPoints = 12;
        int earnedPoints = 0;

        //System.out.println("Results:");
        ArrayList<Integer> res = new ArrayList<>();
        for (Result resType : Result.values()) {
            Integer nOccurrence = iterationOutcomes.get(resType);
            earnedPoints += nOccurrence * pointDict.get(resType);

            res.add(nOccurrence);
            //System.out.println("\t" + resType.toString() + " : " + nOccurrence + " -> " + nOccurrence * pointDict.get(resType) + " points");
        }

        double pointFraction = min((double) earnedPoints / (double) maxPoints, 1.0);

        //System.out.println("{\"fraction\": " + pointFraction + "}");

        res.add(earnedPoints);
        return res;
    }
}