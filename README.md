# My Website Builder

This project is a custom website builder that utilizes Jinja templates and supports Markdown input. It is styled using the Bulma CSS framework to ensure a modern and responsive design.

## Features

- **Dynamic Content Rendering**: The website builder allows users to create web pages with dynamic content using Jinja templates.
- **Markdown Support**: Users can input content in Markdown format, which will be processed and rendered on the web pages.
- **Responsive Design**: The application is styled with the Bulma CSS framework, providing a clean and responsive layout.

## Project Structure

```
my-website-builder
├── src
│   ├── templates
│   │   └── base.jinja       # Base Jinja template for the website
│   ├── static
│   │   └── css
│   │       └── bulma.min.css # Bulma CSS framework for styling
│   ├── app.py               # Main application file
│   └── types
│       └── index.py         # Custom types and data structures
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
└── config.py                 # Configuration settings
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-website-builder
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/app.py
   ```

4. Open your web browser and navigate to `http://localhost:5000` to view the website builder.

## Usage Guidelines

- To create a new page, input your content in Markdown format.
- The content will be dynamically rendered using the base Jinja template.
- Customize the styles by modifying the Bulma CSS framework or adding your own styles.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.