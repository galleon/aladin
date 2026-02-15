/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  safelist: [
    'bg-cyan-500/10', 'text-cyan-400',
    'bg-amber-500/10', 'text-amber-400',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}

