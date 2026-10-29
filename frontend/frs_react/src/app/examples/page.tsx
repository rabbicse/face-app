import Link from "next/link";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const EXAMPLES = [
    {
        href: "/examples/pipeline",
        title: "Live pipeline inspector",
        description: "Webcam -> landmarks -> head pose -> aligned crop -> 512-d embedding, with timings, quality metrics and live similarity to a reference capture.",
    },
    {
        href: "/examples/compare",
        title: "Compare two photos",
        description: "Pick two images; both embeddings are computed in your browser and compared with cosine similarity. Nothing is uploaded.",
    },
    {
        href: "/face/registration?plan=quick&fullname=Demo&email=demo@example.com",
        title: "Quick registration",
        description: "Frontal only enrolment (3 samples) - handy for kiosks or for testing with a static camera.",
    },
];

export default function ExamplesPage() {
    return (
        <main className="mx-auto max-w-3xl space-y-4 p-6">
            <h1 className="text-2xl font-bold">On-device face recognition examples</h1>
            <div className="grid gap-4">
                {EXAMPLES.map(e => (
                    <Link key={e.href} href={e.href}>
                        <Card className="transition hover:shadow-md">
                            <CardHeader>
                                <CardTitle>{e.title}</CardTitle>
                                <CardDescription>{e.description}</CardDescription>
                            </CardHeader>
                        </Card>
                    </Link>
                ))}
            </div>
        </main>
    );
}
