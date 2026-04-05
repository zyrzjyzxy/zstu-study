package Test;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import java.util.Arrays;
import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class DateUtilsTest {
    private int year;
    private int month;
    private int day;
    private String expected;

    // 构造函数，接收测试参数
    public DateUtilsTest(int year, int month, int day, String expected) {
        this.year = year;
        this.month = month;
        this.day = day;
        this.expected = expected;
    }

    // 定义测试数据（边界值分析法）
    @Parameters(name = "{index}: {0}-{1}-{2} → {3}")
    public static Iterable<Object[]> data() {
        return Arrays.asList(new Object[][] {
                {2000, 1, 31, "2000-02-01"},   // 1月最后一天 → 跨月
                {2000, 12, 31, "2001-01-01"},  // 12月最后一天 → 跨年
                {2000, 2, 29, "2000-03-01"},   // 闰年2月结束
                {1900, 2, 28, "1900-03-01"},   // 非闰年2月结束
                {1900, 12, 31, "1901-01-01"},  // 最小年份跨年
                {2050, 12, 31, "2051-01-01"}   // 最大年份跨年
        });
    }

    // 测试方法
    @Test
    public void testBoundaryValues() {
        String actual = DateUtils.getNextDate(year, month, day);
        assertEquals(expected, actual);
    }
}