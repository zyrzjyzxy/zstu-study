import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.By;
import org.openqa.selenium.Dimension;
import org.openqa.selenium.interactions.Actions;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class Example {

    // Mooctest Selenium Example

    // <!> Check if selenium-standalone.jar is added to build path.

    public static void test(WebDriver driver) {
        Map<String, Object> vars = new HashMap<>();
        
        // 访问乐视网站
        driver.get("https://www.le.com/");
        driver.manage().window().setSize(new Dimension(866, 1073));
        
        // 悬停在焦点区域的第一个元素
        {
            WebElement element = driver.findElement(By.cssSelector(".focus_list li:nth-child(1) > a"));
            Actions builder = new Actions(driver);
            builder.moveToElement(element).perform();
        }
        
        // 点击电影链接并等待新窗口
        vars.put("window_handles", driver.getWindowHandles());
        driver.findElement(By.linkText("电影")).click();
        vars.put("win6006", waitForWindow(driver, vars, 2000));
        driver.switchTo().window(vars.get("win6006").toString());
        
        // 点击当前导航项并等待新窗口
        vars.put("window_handles", driver.getWindowHandles());
        driver.findElement(By.cssSelector(".nav_box2 .curr")).click();
        vars.put("win3153", waitForWindow(driver, vars, 2000));
        driver.switchTo().window(vars.get("win3153").toString());
        
        // 选择电影筛选条件
        driver.findElement(By.linkText("免费")).click();
        driver.findElement(By.linkText("动作")).click();
        driver.findElement(By.linkText("中国香港")).click();
        driver.findElement(By.linkText("2016")).click();
        driver.findElement(By.linkText("最新")).click();
        
        // 悬停在第一个视频上
        {
            WebElement element = driver.findElement(By.cssSelector(".videoBox:nth-child(1) img"));
            Actions builder = new Actions(driver);
            builder.moveToElement(element).perform();
        }
        
        // 点击"赏金猎人"电影
        vars.put("window_handles", driver.getWindowHandles());
        driver.findElement(By.xpath("//img[@alt='赏金猎人']")).click();
        vars.put("win6182", waitForWindow(driver, vars, 2000));
        
        {
            WebElement element = driver.findElement(By.tagName("body"));
            Actions builder = new Actions(driver);
            builder.moveToElement(element, 0, 0).perform();
        }
        
        // 切换到电影详情页
        driver.switchTo().window(vars.get("win6182").toString());
        {
            WebElement element = driver.findElement(By.cssSelector(".j-juji-recommend i"));
            Actions builder = new Actions(driver);
            builder.moveToElement(element).perform();
        }
        
        {
            WebElement element = driver.findElement(By.tagName("body"));
            Actions builder = new Actions(driver);
            builder.moveToElement(element, 0, 0).perform();
        }
        
        // 搜索周星驰
        driver.findElement(By.name("wd")).click();
        {
            WebElement element = driver.findElement(By.cssSelector(".search_btn"));
            Actions builder = new Actions(driver);
            builder.moveToElement(element).perform();
        }
        driver.findElement(By.name("wd")).sendKeys("周星驰");
        driver.findElement(By.cssSelector(".search_btn")).click();
        
        // 返回主页
        driver.findElement(By.cssSelector(".channel_home > .icon_text")).click();
        
        // 点击电视剧链接
        vars.put("window_handles", driver.getWindowHandles());
        driver.findElement(By.linkText("电视剧")).click();
        vars.put("win5138", waitForWindow(driver, vars, 2000));
        driver.switchTo().window(vars.get("win5138").toString());
        
        // 点击更多
        vars.put("window_handles", driver.getWindowHandles());
        driver.findElement(By.linkText("更多")).click();
        vars.put("win1590", waitForWindow(driver, vars, 2000));
        driver.switchTo().window(vars.get("win1590").toString());
        
        // 选择电视剧筛选条件
        driver.findElement(By.linkText("独播")).click();
        driver.findElement(By.linkText("穿越")).click();
        driver.findElement(By.linkText("2013")).click();
        
        // 悬停在图片上
        {
            WebElement element = driver.findElement(By.cssSelector(".imagePart > img"));
            Actions builder = new Actions(driver);
            builder.moveToElement(element).perform();
        }
        
        // 点击剧集图片
        vars.put("window_handles", driver.getWindowHandles());
        driver.findElement(By.cssSelector(".imagePart > img")).click();
        vars.put("win9151", waitForWindow(driver, vars, 2000));
        
        {
            WebElement element = driver.findElement(By.tagName("body"));
            Actions builder = new Actions(driver);
            builder.moveToElement(element, 0, 0).perform();
        }
        
        // 切换到剧集详情页
        driver.switchTo().window(vars.get("win9151").toString());
        
        // 点击相关标签
        driver.findElement(By.linkText("花絮")).click();
        driver.findElement(By.linkText("分集剧情")).click();
        driver.findElement(By.linkText("乐迷畅谈")).click();
        
        // 最后悬停在标题上
        {
            WebElement element = driver.findElement(By.cssSelector("h2 > em"));
            Actions builder = new Actions(driver);
            builder.moveToElement(element).perform();
        }
    }
    
    // 辅助方法：等待新窗口打开
    private static String waitForWindow(WebDriver driver, Map<String, Object> vars, int timeout) {
        try {
            Thread.sleep(timeout);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Set<String> whNow = driver.getWindowHandles();
        Set<String> whThen = (Set<String>) vars.get("window_handles");
        if (whNow.size() > whThen.size()) {
            whNow.removeAll(whThen);
        }
        return whNow.iterator().next();
    }

    public static void main(String[] args) {
        System.setProperty("webdriver.chrome.driver", "C:/Program Files/Google/Chrome/Application/chromedriver-win64/chromedriver.exe");
        // Run main function to test your script.
        WebDriver driver = new ChromeDriver();
        try { test(driver); } 
        catch(Exception e) { e.printStackTrace(); }
        finally { driver.quit(); }
    }
}