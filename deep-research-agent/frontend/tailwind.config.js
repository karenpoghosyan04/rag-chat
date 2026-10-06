/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#17211b',
        paper: '#f6f8f5',
        moss: '#315c47',
        line: '#dbe2dc',
      },
      fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'] },
      boxShadow: { soft: '0 16px 50px rgba(27, 53, 40, 0.08)' },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}

