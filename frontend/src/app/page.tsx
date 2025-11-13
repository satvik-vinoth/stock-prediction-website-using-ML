"use client";

import { useState } from "react";
import Header from "@/components/header";
import FrontPage from "@/components/frontpage";
import dynamic from "next/dynamic";
const CompanySelector = dynamic(() => import("@/components/CompanySelector"), { ssr: false });
import ModelSelector from "@/components/ModelSelector";
import { useEffect } from "react";


export default function Home() {
  const [selectedCompany, setSelectedCompany] = useState("AAPL");
  const [backendReady, setBackendReady] = useState(false);
  const [checking, setChecking] = useState(true);
  const baseurl = process.env.NEXT_PUBLIC_API_BASE_URL

  useEffect(() => {
    const checkBackend = async () => {
      try {
        console.log(baseurl)
        const res = await fetch(`${baseurl}/health`);
        if (res.ok) {
          setBackendReady(true);
          setChecking(false);
          return;
        }
      } catch  {
        
      }
      setTimeout(checkBackend, 3000); 
    };

    checkBackend();
  }, []);

  if (!backendReady && checking) {
    return (
      <div className="bg-[#1c3b35] min-h-screen flex items-center justify-center text-white text-center p-6">
        <div>
          <h2 className="text-2xl font-semibold">Waking up the backend...</h2>
          <p className="mt-4 text-sm text-gray-300">
            Due to free-tier limitations, the backend may take 2-3 minutes to wake up. Please wait...
          </p>
        </div>
      </div>
    );
  }

  return (
    
    <div className="bg-[#1c3b35] min-h-screen relative overflow-hidden">
      <section id="home">
      </section>
      <Header/>
      <FrontPage/>
      <section id="company">
      </section>
      <CompanySelector onCompanySelected={setSelectedCompany} />
      <section id="prediction">
      </section>
      <ModelSelector company={selectedCompany} />

      <footer className="w-full text-center text-sm text-gray-500 py-8" id="contact">
        &copy; {new Date().getFullYear()} Stock Vision -1234567890. All rights reserved.
      </footer>
    </div>
  );
}
