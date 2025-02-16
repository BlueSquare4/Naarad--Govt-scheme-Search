# **Naarad - Personalized Government Schemes Finder**

Naarad is a user-friendly website that helps individuals discover government schemes tailored to their specific needs based on **income**, **occupation**, and **location**. It simplifies access to welfare programs, empowering users to make the most of available resources.

---

## **Features**

- **Custom Scheme Filtering**: Users can find schemes based on their:
  - **Income Range** (e.g., below ₹2,00,000, ₹2,50,000, etc.)
  - **Occupation** (e.g., farming, urban professional)
  - **Location** (rural or urban)
- **Dynamic Display**: Schemes are dynamically updated and displayed based on user parameters.
- **Detailed Information**: Each scheme includes a description and a direct link to the official government website for more details.

---

## **How It Works**

1. Users pass parameters via the URL, such as `income`, `occupation`, and `location`.
2. Naarad filters relevant schemes based on these parameters.
3. The website displays a personalized list of schemes with descriptions and links.

### Example URL:
```
index.html?income=250000&occupation=farming&location=rural
```

---

## **Technologies Used**

- **HTML5** and **CSS3**: For the responsive and clean front-end design.
- **JavaScript (ES6)**: To dynamically filter and display schemes.
- **Government Resources**: Official scheme links for authentic information.

---

## **Usage**

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/naarad.git
   ```
2. Navigate to the project directory:  
   ```bash
   cd naarad
   ```
3. Open `index.html` in your browser.
4. Pass query parameters (e.g., `?income=250000&occupation=farming&location=rural`) to explore the tailored schemes.

---

## **Contributing**

Contributions are welcome! Here's how you can help:

1. Fork the repository.
2. Create a new branch:  
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:  
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to the branch:  
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

---

## **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---


## **Acknowledgments**

- Government of India for providing detailed scheme information.
- The community for supporting open-source initiatives like Naarad.

---
