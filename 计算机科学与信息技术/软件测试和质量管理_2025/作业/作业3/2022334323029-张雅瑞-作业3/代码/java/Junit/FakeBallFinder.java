package Junit;

import java.util.Arrays;
import java.util.Random;

public class FakeBallFinder {

    public static void main(String[] args) {
        // 创建10个铅球，重量都为100，其中一个设为假球(重量为90)
        int[] balls = new int[10];
        Arrays.fill(balls, 100);

        Random random = new Random();
        int fakeBallIndex = random.nextInt(10);
        balls[fakeBallIndex] = 90;

        System.out.println("铅球重量数组: " + Arrays.toString(balls));
        System.out.println("实际假球位置: " + (fakeBallIndex + 1));

        // 开始称重
        int result = findFakeBall(balls);
        System.out.println("找到的假球位置: " + (result + 1));
    }

    public static int findFakeBall(int[] balls) {
        // 第一次称重：分成两组，每组5个球
        int leftSum1 = 0, rightSum1 = 0;
        for (int i = 0; i < 5; i++) {
            leftSum1 += balls[i];
            rightSum1 += balls[i + 5];
        }

        int[] candidateBalls;
        if (leftSum1 < rightSum1) {
            System.out.println("第一次称重: 左边较轻");
            candidateBalls = Arrays.copyOfRange(balls, 0, 5);
        } else if (leftSum1 > rightSum1) {
            System.out.println("第一次称重: 右边较轻");
            candidateBalls = Arrays.copyOfRange(balls, 5, 10);
        } else {
            // 这种情况理论上不会发生，因为题目保证有一个假球
            System.out.println("第一次称重: 两边重量相同");
            return -1;
        }

        // 第二次称重：从候选的5个球中取4个，分成两组，每组2个
        int leftSum2 = candidateBalls[0] + candidateBalls[1];
        int rightSum2 = candidateBalls[2] + candidateBalls[3];

        if (leftSum2 == rightSum2) {
            System.out.println("第二次称重: 两边重量相同，剩下的是假球");
            // 剩下的是假球(第5个)
            return findOriginalIndex(balls, candidateBalls[4]);
        } else if (leftSum2 < rightSum2) {
            System.out.println("第二次称重: 左边较轻");
            // 假球在左边两个中
            return thirdWeighing(balls, candidateBalls[0], candidateBalls[1]);
        } else {
            System.out.println("第二次称重: 右边较轻");
            // 假球在右边两个中
            return thirdWeighing(balls, candidateBalls[2], candidateBalls[3]);
        }
    }

    private static int thirdWeighing(int[] allBalls, int ball1, int ball2) {
        System.out.println("第三次称重: 比较两个候选球");
        if (ball1 < ball2) {
            return findOriginalIndex(allBalls, ball1);
        } else {
            return findOriginalIndex(allBalls, ball2);
        }
    }

    private static int findOriginalIndex(int[] allBalls, int ball) {
        for (int i = 0; i < allBalls.length; i++) {
            if (allBalls[i] == ball) {
                return i;
            }
        }
        return -1;
    }
}