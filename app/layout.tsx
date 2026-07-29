import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "hoopvision.ai";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.includes("localhost") ? "http" : "https");
  const baseUrl = `${protocol}://${host}`;

  return {
    title: "HoopVision AI - Live AI Coach",
    description: "Live AI basketball coaching with motion analysis, instant form feedback, voice cues, automatic skill tracking, game-film intelligence, pro comparison, and player development reports.",
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
    openGraph: {
      title: "HoopVision AI - Live AI Coach",
      description: "Live movement coaching, skill tracking, Film IQ, pro comparison, and personalized player development.",
      type: "website",
      images: [{ url: `${baseUrl}/og.png`, width: 1200, height: 630, alt: "HoopVision AI live movement coaching interface" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "HoopVision AI - Live AI Coach",
      description: "Live movement coaching, skill tracking, Film IQ, pro comparison, and personalized player development.",
      images: [`${baseUrl}/og.png`],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
