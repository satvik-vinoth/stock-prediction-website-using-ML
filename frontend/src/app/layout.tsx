import './globals.css';
import { manrope } from '@/lib/font';

export const metadata = {
  title: 'Stock Vision',
  description: 'AI-powered stock forecasting',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={manrope.className}>
        {children}
      </body>
    </html>
  );
}
