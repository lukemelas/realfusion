import { Heading, Wrap, Link as ChakraLink } from '@chakra-ui/react'
import { Container } from 'next/app'
import NextLink from "next/link"

const institutions = {
  1: "Oxford University",
  2: "Harvard University",
}

const authors = [
  {
    'name': 'Luke Melas-Kyraizi',
    'institutions': [1, 2],
    'url': "https://lukemelas.github.io/"
  },
]

export const Hero = ({ title }: { title: string }) => (
  <Container>
    <Heading fontSize="3xl" pt="3rem">{title}</Heading>
    <Wrap justify="center" pt="1rem" fontSize="xl" key="authors">
      {
        authors.map((author) =>
          <span key={author.name}>
            <NextLink href={author.url} passHref={true}>
              <ChakraLink>{author.name}</ChakraLink>
            </NextLink>
            <sup> {author.institutions.toString()}</sup>
          </span>
        )
      }
    </Wrap>
    <Wrap justify="center" pt="1rem" key="institutions">
      {
        Object.entries(institutions).map(tuple =>
          <span>
            <sup>{tuple[0]}  </sup>
            {tuple[1]}
          </span>
        )
      }
    </Wrap>
  </Container>
)

Hero.defaultProps = {
  title: 'Academic Project Template',
}