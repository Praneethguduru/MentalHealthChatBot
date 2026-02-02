"use client";
import { useState } from "react";
import { supabase } from "@/lib/supabase";
import { useRouter } from "next/navigation";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();

  const handleAuth = async (isLogin: boolean) => {
    const { error } = isLogin 
      ? await supabase.auth.signInWithPassword({ email, password })
      : await supabase.auth.signUp({ email, password });
    if (error) alert(error.message);
    else router.push("/chat");
  };

  return (
    <div className="h-screen flex items-center justify-center bg-blue-50">
      <div className="bg-white p-6 rounded-xl shadow-lg w-80 flex flex-col gap-4">
        <h1 className="text-xl font-bold text-center">Mental Health AI</h1>
        <input className="p-2 border rounded" placeholder="Email" onChange={e => setEmail(e.target.value)}/>
        <input className="p-2 border rounded" type="password" placeholder="Password" onChange={e => setPassword(e.target.value)}/>
        <button onClick={() => handleAuth(true)} className="bg-blue-600 text-white p-2 rounded">Login</button>
        <button onClick={() => handleAuth(false)} className="text-sm text-blue-600">Sign Up</button>
      </div>
    </div>
  );
}