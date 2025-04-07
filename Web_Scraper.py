from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from urllib.parse import urljoin


def scrape_shl_products():
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Optional: Run in background
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    # Set up driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    base_url = "https://www.shl.com"
    catalog_url = "https://www.shl.com/solutions/products/product-catalog/"

    try:
        print("Loading SHL product catalog...")
        driver.get(catalog_url)

        # Wait for products to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".product-card"))
        )

        # Scroll to load all products
        print("Scrolling to load all products...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        product_cards = driver.find_elements(By.CSS_SELECTOR, ".product-card")
        print(f"Found {len(product_cards)} products.")

        products = []

        for card in product_cards:
            try:
                product = {
                    'Assessment Name': 'Not found',
                    'URL': 'Not found',
                    'Remote Testing Support': 'No',
                    'Adaptive/IRT Support': 'No',
                    'Duration': 'Not specified',
                    'Test Type': 'Not specified'
                }

                # Name
                name_element = card.find_element(By.CSS_SELECTOR, ".product-card__title")
                product['Assessment Name'] = name_element.text

                # URL
                link_element = card.find_element(By.CSS_SELECTOR, "a[href]")
                product['URL'] = urljoin(base_url, link_element.get_attribute("href"))

                # Metadata
                meta_items = card.find_elements(By.CSS_SELECTOR, ".product-card__meta-item")
                for item in meta_items:
                    try:
                        label = item.find_element(By.CSS_SELECTOR, ".product-card__meta-label").text.lower()
                        value = item.find_element(By.CSS_SELECTOR, ".product-card__meta-value").text

                        if 'remote' in label:
                            product['Remote Testing Support'] = 'Yes' if 'yes' in value.lower() else 'No'
                        elif 'adaptive' in label or 'irt' in label:
                            product['Adaptive/IRT Support'] = 'Yes' if 'yes' in value.lower() else 'No'
                        elif 'duration' in label:
                            product['Duration'] = value
                        elif 'type' in label:
                            product['Test Type'] = value
                    except NoSuchElementException:
                        continue

                products.append(product)

            except Exception as e:
                print(f"Error processing a product card: {str(e)}")
                continue

        # Save data
        df = pd.DataFrame(products)
        df.to_csv('shl_products.csv', index=False)
        print("Data saved to shl_products.csv")

        return df

    except TimeoutException:
        print("Timeout loading the page.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        driver.quit()
        print("Browser closed.")


if __name__ == "__main__":
    print("Starting SHL scraper...")   # Debug print
    df = scrape_shl_products()
    if df is not None and not df.empty:
        print("\nFirst 5 results:")
        print(df.head())
    else:
        print("No data scraped.")
