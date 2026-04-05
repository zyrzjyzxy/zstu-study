package Test;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Arrays;


public class FakeBallFinderTest {

    // Helper method to create test balls with fake at specified index
    private int[] createTestBalls(int fakeIndex) {
        int[] balls = new int[10];
        Arrays.fill(balls, 100);
        balls[fakeIndex] = 90;
        return balls;
    }

    /* 第一次称重左边轻的测试用例 */

    @Test
    public void testLeftGroup_FakeIsRemainingBall() {
        // 路径1: 第一次左边轻 → 第二次称重等重 → 返回剩下的第五个球(索引4)
        int[] balls = createTestBalls(4);
        assertEquals(4, FakeBallFinder.findFakeBall(balls));
    }

    @Test
    public void testLeftGroup_FakeInFirstTwo_FirstIsFake() {
        // 路径2: 第一次左边轻 → 第二次左边轻 → 第三次返回左边球(索引0)
        int[] balls = createTestBalls(0);
        assertEquals(0, FakeBallFinder.findFakeBall(balls));
    }

    @Test
    public void testLeftGroup_FakeInFirstTwo_SecondIsFake() {
        // 路径3: 第一次左边轻 → 第二次左边轻 → 第三次返回右边球(索引1)
        int[] balls = createTestBalls(1);
        assertEquals(1, FakeBallFinder.findFakeBall(balls));
    }

    @Test
    public void testLeftGroup_FakeInNextTwo_FirstIsFake() {
        // 路径4: 第一次左边轻 → 第二次右边轻 → 第三次返回左边球(索引2)
        int[] balls = createTestBalls(2);
        assertEquals(2, FakeBallFinder.findFakeBall(balls));
    }

    @Test
    public void testLeftGroup_FakeInNextTwo_SecondIsFake() {
        // 路径5: 第一次左边轻 → 第二次右边轻 → 第三次返回右边球(索引3)
        int[] balls = createTestBalls(3);
        assertEquals(3, FakeBallFinder.findFakeBall(balls));
    }

    /* 第一次称重右边轻的测试用例 */

    @Test
    public void testRightGroup_FakeIsRemainingBall() {
        // 路径6: 第一次右边轻 → 第二次称重等重 → 返回剩下的第五个球(索引9)
        int[] balls = createTestBalls(9);
        assertEquals(9, FakeBallFinder.findFakeBall(balls));
    }

    @Test
    public void testRightGroup_FakeInFirstTwo_FirstIsFake() {
        // 路径7: 第一次右边轻 → 第二次左边轻 → 第三次返回左边球(索引5)
        int[] balls = createTestBalls(5);
        assertEquals(5, FakeBallFinder.findFakeBall(balls));
    }

    @Test
    public void testRightGroup_FakeInFirstTwo_SecondIsFake() {
        // 路径8: 第一次右边轻 → 第二次左边轻 → 第三次返回右边球(索引6)
        int[] balls = createTestBalls(6);
        assertEquals(6, FakeBallFinder.findFakeBall(balls));
    }

    @Test
    public void testRightGroup_FakeInNextTwo_FirstIsFake() {
        // 路径9: 第一次右边轻 → 第二次右边轻 → 第三次返回左边球(索引7)
        int[] balls = createTestBalls(7);
        assertEquals(7, FakeBallFinder.findFakeBall(balls));
    }

    @Test
    public void testRightGroup_FakeInNextTwo_SecondIsFake() {
        // 路径10: 第一次右边轻 → 第二次右边轻 → 第三次返回右边球(索引8)
        int[] balls = createTestBalls(8);
        assertEquals(8, FakeBallFinder.findFakeBall(balls));
    }

    /* 异常情况测试 */

    @Test
    public void testNoFakeBall() {
        // 路径11: 第一次称重等重 → 返回-1(异常情况)
        int[] balls = new int[10];
        Arrays.fill(balls, 100);
        assertEquals(-1, FakeBallFinder.findFakeBall(balls));
    }
}