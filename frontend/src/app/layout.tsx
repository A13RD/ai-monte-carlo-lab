import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Monte Carlo Lab",
  description:
    "Explora la incertidumbre de demanda, consumo de tokens y costos de inferencia con simulaciones reproducibles.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
