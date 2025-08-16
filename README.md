# Water Content App

## Prerequisites

Node.js (version 14 or higher)
Python (version 3.8 or higher)
pip (Python package manager)
npm (Node package manager)

## Frontend Setup

In any IDE,open the project folder and navigate to the project directory root in a terminal.

Install dependencies by typing : `npm install` in the terminal

Start the development server with : `npm run dev`

The frontend should now be running at http://localhost:5173. Got to this URL in any browser to see the site. (If in Chrome, go to View -> Developer -> Developer Tools, then click the devices icon in the upper left corner panel to change to mobile device.)

## Backend Setup

In a new terminal, navigate to the api directory by typing in: `cd api`

Create a virtual environment: `python -m venv venv`

Activate the virtual environment:
On Windows: `venv\Scripts\activate`
On macOS/Linux: `source venv/bin/activate`

Install dependencies: `pip install -r requirements.txt`
Start the development server: `python app.py` or `flask run`

The backend should now be running at http://127.0.0.1:5000.

## Notes

Make sure both the backend and frontend are running properly before going to the site.
Make sure to update the getApiUrl() function in the frontend code to point to the correct backend URL (http://127.0.0.1:5000) when running locally.
